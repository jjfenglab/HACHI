import asyncio
import logging
import os
import pickle
import sys
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.append(os.getcwd())
sys.path.append("llm-api-main")
from lab_llm.dataset import ImageDataset, TextDataset
from lab_llm.llm_api import LLMApi

from src.llm_response_types import ExtractResponseList, GroupedExtractResponses
from src.logistic import LogisticRegressionTorch

TABULAR_PREFIX = "Tabular feature: "


def is_tabular(concept):
    return concept.startswith(TABULAR_PREFIX)


def get_features(
    concept_dicts,
    all_extracted_features,
    dset,
    force_keep_columns: Optional[List[str]] = None,
):
    concept_vals = []
    for concept_dict in concept_dicts:
        concept = concept_dict["concept"]
        concept_vals.append(all_extracted_features[concept])
    if force_keep_columns is not None:
        # Scale the control features
        scaler = StandardScaler()
        control_features = scaler.fit_transform(dset[force_keep_columns])
        concept_vals = [control_features] + concept_vals
    return np.concatenate(concept_vals, axis=1)


def split_sentences_by_id(
    dset_train: pd.DataFrame,
    max_section_length: Union[int, None] = None,
    sentence_column: str = "sentence",
) -> tuple[np.array, np.array]:
    """
    If texts are too long, split it into sections and return group_id as well as the sections
    """
    if max_section_length is None:
        return np.arange(dset_train.shape[0]), dset_train.sentence.to_numpy()
    else:
        sentences = []
        ids = []
        for idx, row in dset_train.iterrows():
            sentence = row[sentence_column]
            if len(sentence) > max_section_length:
                for start in range(0, len(sentence), max_section_length):
                    end = min(start + max_section_length, len(sentence))
                    ids.append(idx)
                    sentences.append(sentence[start:end])
            else:
                sentences.append(sentence)
                ids.append(idx)
        return np.array(ids), np.array(sentences)


def _collate_extractions_by_group(
    llm_outputs: List, group_ids, num_to_extract: int = 6
) -> np.ndarray:
    """
    Combines the extracted feature vectors if they have the same group ids
    @param num_to_extract: this is the number of concepts in a batch, assumes you are extracting the same number of features per concept-batch
    """
    extracted_llm_outputs = []
    for grp_id in np.unique(group_ids):
        match_idxs = np.where(group_ids == grp_id)[0]
        id_llm_outputs = np.concatenate(
            [
                _extract_features(
                    llm_outputs[match_idx], num_concepts=num_to_extract
                ).reshape((1, -1))
                for match_idx in match_idxs
            ],
            axis=0,
        )

        assert id_llm_outputs.shape == (len(match_idxs), num_to_extract)
        concepts_for_id = np.max(id_llm_outputs, axis=0)
        assert concepts_for_id.shape == (num_to_extract,)
        extracted_llm_outputs.append(concepts_for_id)
    extracted_llm_outputs = np.vstack(extracted_llm_outputs)
    return extracted_llm_outputs


def _extract_features(llm_output: ExtractResponseList, num_concepts: int):
    features_output = [0] * num_concepts
    if llm_output is not None:
        for extraction_resp in llm_output.extractions:
            if extraction_resp.question <= num_concepts:
                features_output[(extraction_resp.question - 1)] = extraction_resp.answer
            else:
                logging.warning(
                    "question ID does not exist... %d %s",
                    extraction_resp.question,
                    extraction_resp.reasoning,
                )
    return np.array(features_output)


def get_auc_and_probs(mdl, X, y, sample_weight) -> tuple[float, list]:
    is_multi_class = len(np.unique(y)) > 2
    if is_multi_class:
        classes = np.unique(y)
        y_pred = mdl.predict_proba(X)
        y = label_binarize(y, classes=classes)
        auc = roc_auc_score(y, y_pred, sample_weight=sample_weight, multi_class="ovr")
    else:
        y_pred = mdl.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred, sample_weight=sample_weight)

    return auc, y_pred


# to avoid numerical underflow/overflow


def get_safe_prob(y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return y_pred


def get_safe_logit(y_pred, eps=1e-15):
    y_pred = get_safe_prob(y_pred, eps)
    return np.log(y_pred / (1 - y_pred))


def get_log_liks(y, pred_prob, is_multiclass=False):
    safe_pred_prob = get_safe_prob(pred_prob)
    if not is_multiclass:
        # binary
        return y * np.log(safe_pred_prob) + (1 - y) * np.log(1 - safe_pred_prob)
    else:
        # multiclass
        return np.log(safe_pred_prob[np.arange(safe_pred_prob.shape[0]), y])


def train_LR(
    X_train,
    y_train,
    sample_weight,
    X_test=None,
    y_test=None,
    test_weight=None,
    penalty=None,
    use_acc: bool = False,
    seed: int = 0,
    cv: int = 5,
) -> dict:
    _, class_counts = np.unique(y_train, return_counts=True)
    assert class_counts.size == 2
    cv = 2 if np.any(class_counts < 3) else cv
    is_multi_class = len(np.unique(y_train)) > 2
    args = {
        "cv": StratifiedKFold(n_splits=cv),
        "scoring": (
            "accuracy" if use_acc else ("roc_auc_ovr" if is_multi_class else "roc_auc")
        ),
        "n_jobs": -1,
    }

    if penalty is None:
        final_model = LogisticRegression(penalty=None, random_state=seed)
        final_model.fit(X_train, y_train, sample_weight=sample_weight)
    # elif penalty == "l1":
    #     param_grid = {"lambd": [0.01, 0.001, 0.0001, 1e-5], "num_epochs": [7000]}
    #     model = GridSearchCV(
    #         estimator=LogisticRegressionTorch(seed=seed), param_grid=param_grid, **args
    #     )
    #     model.fit(X_train, y_train)
    #     print("CV SCORES ACC/AUC", model.best_score_)
    #     logging.info("CV SCORES best_score_ %s", model.best_score_)
    #     logging.info("CV RESULTS %s", model.cv_results_)
    #     final_model = model.best_estimator_
    #     logging.info(
    #         "NUM NONZERO 1e-2 %d",
    #         np.sum(~np.isclose(np.abs(final_model.coef_).max(axis=0), 0, atol=1e-2)),
    #     )
    elif penalty == "l1_sklearn":
        final_model = LogisticRegressionCV(
            max_iter=10000,
            Cs=10,
            solver="saga",
            penalty="l1",
            random_state=seed,
            **args,
        )
        final_model.fit(X_train, y_train, sample_weight=sample_weight)
    elif penalty == "l2":
        final_model = LogisticRegressionCV(
            max_iter=10000, Cs=10, solver="lbfgs", random_state=seed, **args
        )
        final_model.fit(X_train, y_train, sample_weight=sample_weight)

    train_auc, y_pred = get_auc_and_probs(
        final_model, X_train, y_train, sample_weight=sample_weight
    )
    y_assigned_class = final_model.predict(X_train)
    train_logit = get_safe_logit(y_pred)
    res_dict = {
        "model": final_model,
        "auc": train_auc,  # train auc
        "logit": train_logit,
        "coef": final_model.coef_,
        "intercept": final_model.intercept_,
        "y_pred": y_pred,
        "y_assigned_class": y_assigned_class,
        "acc": accuracy_score(y_train, y_assigned_class),  # train acc
    }
    if X_test is not None:
        test_auc, _ = get_auc_and_probs(
            final_model, X_test, y_test, sample_weight=test_weight
        )
        res_dict["test_auc"] = test_auc
        logging.info(f"test_auc {test_auc}")
        y_assigned_class = final_model.predict(X_test)
        res_dict["test_acc"] = (
            accuracy_score(y_test, y_assigned_class, sample_weight=test_weight),
        )  # train acc

        for weight_uniq in np.unique(test_weight):
            match_idx = test_weight == weight_uniq
            test_domain_auc, _ = get_auc_and_probs(
                final_model,
                X_test[match_idx],
                y_test[match_idx],
                sample_weight=np.ones(np.sum(match_idx)),
            )
            logging.info(f"TEST AUC domain {weight_uniq} {test_domain_auc}")

    return res_dict


def train_LR_max_features(
    X_train,
    y_train,
    sample_weight,
    X_test=None,
    y_test=None,
    test_weight=None,
    num_meta_concepts: int = 5,
    seed: int = 0,
) -> dict:
    _, class_counts = np.unique(y_train, return_counts=True)
    assert class_counts.size == 2

    # Search for appropriate C value to get at most 5 features
    # Smaller C = stronger regularization = more sparsity
    C_candidates = np.logspace(-4, 2, 200)[::-1]  # Try C values from 0.0001 to 100

    best_C = None
    final_model = None

    # rf = RandomForestClassifier()
    # rf.fit(X_train, y_train, sample_weight=sample_weight)
    # test_auc, _ = get_auc_and_probs(rf, X_test, y_test, sample_weight=test_weight)
    # print("RF test auc", test_auc)

    for C in C_candidates:
        model = LogisticRegression(
            penalty="l1", solver="saga", C=C, random_state=seed, max_iter=1000
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        # test_auc, _ = get_auc_and_probs(model, X_test, y_test, test_weight=test_weight)
        n_features = np.sum(model.coef_ != 0)
        # print("LR", C, n_features, test_auc)

        if n_features <= num_meta_concepts:
            best_C = C
            final_model = model
            break

    if final_model is None:
        raise ValueError("cant find a good one")

    train_auc, y_pred = get_auc_and_probs(
        final_model, X_train, y_train, sample_weight=sample_weight
    )
    y_assigned_class = final_model.predict(X_train)
    train_logit = get_safe_logit(y_pred)
    res_dict = {
        "model": final_model,
        "auc": train_auc,  # train auc
        "logit": train_logit,
        "coef": final_model.coef_,
        "intercept": final_model.intercept_,
        "y_pred": y_pred,
        "y_assigned_class": y_assigned_class,
        "acc": accuracy_score(y_train, y_assigned_class),  # train acc
    }
    if X_test is not None:
        test_auc, _ = get_auc_and_probs(
            final_model, X_test, y_test, sample_weight=test_weight
        )
        res_dict["test_auc"] = test_auc
        logging.info(f"test_auc {test_auc}")
        print(f"test_auc {test_auc}")
    return res_dict


def extract_features_by_llm_grouped(
    llm: LLMApi,
    dset_train,
    meta_concept_dicts: list[dict],
    prompt_file,
    all_extracted_features_dict: dict = dict(),
    extraction_file: str = None,
    batch_size=1,
    batch_concept_size=20,
    max_new_tokens=5000,
    is_image=False,
    group_size: int = 1,
    max_retries: int = 1,
    max_section_length: int = None,
    sentence_column: str = "sentence",
) -> dict[str, np.ndarray]:
    prior_concepts = set(all_extracted_features_dict.keys())
    logging.info(f"all_extracted_features_dict {prior_concepts}")
    concepts_to_extract = []
    for concept_dict in meta_concept_dicts:
        concept = concept_dict["concept"]
        if (
            not is_tabular(concept)
            and concept not in concepts_to_extract
            and concept not in prior_concepts
        ):
            concepts_to_extract.append(concept)

    logging.info(f"LEN CONCEPTS {len(concepts_to_extract)}")

    logging.info(f"dset_train {dset_train.shape}")
    for i in range(0, len(concepts_to_extract), batch_concept_size):
        # Get the batch of concepts to annotate
        batch_concepts_to_extract = concepts_to_extract[i : i + batch_concept_size]

        # Load prompt for extracting concepts
        prompt_file = os.path.abspath(os.path.join(os.getcwd(), prompt_file))
        with open(prompt_file, "r") as file:
            prompt_template = file.read()

        # Fill in concept questions
        prompt_questions = ""
        for idx, concept in enumerate(batch_concepts_to_extract):
            prompt_questions += f"{idx + 1} - {concept}" + "\n"
        prompt_template = prompt_template.replace(
            "{prompt_questions}", prompt_questions
        )
        logging.debug(prompt_template)

        logging.info(f"dset_train.shape[0] {dset_train.shape[0]}")
        if group_size > 1:
            # group together observations for extractions
            obs_group_idxs = []
            if is_image:
                group_ids = np.arange(dset_train.shape[0])
                image_paths_list = []
                for i in range(0, dset_train.shape[0], group_size):
                    group_df = dset_train.image_path.iloc[i : i + group_size]
                    image_paths_list.append(group_df.tolist())
                    obs_group_idxs.append([j for j in range(i, i + group_df.shape[0])])

                dataset = ImageGroupDataset(
                    image_paths_list, prompt_template=prompt_template
                )
            else:
                raise NotImplementedError(
                    "have not yet implemented grouped-extraction of text data"
                )

            grouped_llm_outputs = asyncio.run(
                llm.get_outputs(
                    dataset,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    is_image=is_image,
                    temperature=0,
                    max_retries=max_retries,
                    response_model=GroupedExtractResponses,
                )
            )
            logging.debug(
                f"grouped_llm_outputs {len(grouped_llm_outputs)} {dset_train.shape[0]}"
            )
            # ungroup responses
            llm_outputs = [None] * dset_train.shape[0]
            for group_obs_idxs, grouped_output in zip(
                obs_group_idxs, grouped_llm_outputs
            ):
                if grouped_output is not None:
                    for j, extraction in enumerate(
                        grouped_output.all_extractions[: len(group_obs_idxs)]
                    ):
                        llm_outputs[group_obs_idxs[j]] = extraction
                else:
                    logging.info(
                        f"warning: llm output was missing for idxs {group_obs_idxs}"
                    )
            logging.debug(f"llm_outputs {len(llm_outputs)} {dset_train.shape[0]}")
        else:
            if is_image:
                group_ids = np.arange(dset_train.shape[0])
                dataset = ImageDataset(dset_train.image_path.tolist(), prompt_template)
            else:
                group_ids, sentences = split_sentences_by_id(
                    dset_train, max_section_length, sentence_column
                )
                prompts = [prompt_template.replace("{sentence}", s) for s in sentences]
                dataset = TextDataset(
                    prompts,
                )

            llm_outputs = asyncio.run(
                llm.get_outputs(
                    dataset,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    is_image=is_image,
                    temperature=0,
                    max_retries=max_retries,
                    response_model=ExtractResponseList,
                )
            )

        # extract responses
        extracted_llm_outputs = _collate_extractions_by_group(
            llm_outputs, group_ids, len(batch_concepts_to_extract)
        )

        # fill in dictionary with extractions
        for idx, concept in enumerate(batch_concepts_to_extract):
            extracted_features = extracted_llm_outputs[:, idx : idx + 1]
            logging.info(
                "concept %s, prevalence %f", concept, np.mean(extracted_features)
            )
            all_extracted_features_dict[concept] = extracted_features

        if extraction_file is not None:
            with open(extraction_file, "wb") as f:
                pickle.dump(all_extracted_features_dict, f)

    return all_extracted_features_dict


async def extract_features_by_llm_grouped_async(
    llm: LLMApi,
    dset_train,
    meta_concept_dicts: list[dict],
    prompt_file,
    all_extracted_features_dict: dict = dict(),
    extraction_file: str = None,
    batch_size=1,
    batch_concept_size=20,
    max_new_tokens=5000,
    is_image=False,
    group_size: int = 1,
    max_retries: int = 1,
    max_section_length: int = None,
    sentence_column: str = "sentence",
) -> dict[str, np.ndarray]:
    """
    Async version of extract_features_by_llm_grouped.

    This version uses await instead of asyncio.run() and is designed to be called
    from within an already running event loop (e.g., from async functions).
    """
    prior_concepts = set(all_extracted_features_dict.keys())
    logging.info(f"all_extracted_features_dict {prior_concepts}")
    concepts_to_extract = []
    for concept_dict in meta_concept_dicts:
        concept = concept_dict["concept"]
        if (
            not is_tabular(concept)
            and concept not in concepts_to_extract
            and concept not in prior_concepts
        ):
            concepts_to_extract.append(concept)

    logging.info(f"LEN CONCEPTS {len(concepts_to_extract)}")

    logging.info(f"dset_train {dset_train.shape}")
    for i in range(0, len(concepts_to_extract), batch_concept_size):
        # Get the batch of concepts to annotate
        batch_concepts_to_extract = concepts_to_extract[i : i + batch_concept_size]

        # Load prompt for extracting concepts
        prompt_file = os.path.abspath(os.path.join(os.getcwd(), prompt_file))
        with open(prompt_file, "r") as file:
            prompt_template = file.read()

        # Fill in concept questions
        prompt_questions = ""
        for idx, concept in enumerate(batch_concepts_to_extract):
            prompt_questions += f"{idx + 1} - {concept}" + "\n"
        prompt_template = prompt_template.replace(
            "{prompt_questions}", prompt_questions
        )
        logging.debug(prompt_template)

        logging.info(f"dset_train.shape[0] {dset_train.shape[0]}")
        if group_size > 1:
            # group together observations for extractions
            obs_group_idxs = []
            if is_image:
                group_ids = np.arange(dset_train.shape[0])
                image_paths_list = []
                for i in range(0, dset_train.shape[0], group_size):
                    group_df = dset_train.image_path.iloc[i : i + group_size]
                    image_paths_list.append(group_df.tolist())
                    obs_group_idxs.append([j for j in range(i, i + group_df.shape[0])])

                dataset = ImageGroupDataset(
                    image_paths_list, prompt_template=prompt_template
                )
            else:
                raise NotImplementedError(
                    "have not yet implemented grouped-extraction of text data"
                )

            # Use await instead of asyncio.run()
            grouped_llm_outputs = await llm.get_outputs(
                dataset,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                is_image=is_image,
                temperature=0,
                max_retries=max_retries,
                response_model=GroupedExtractResponses,
            )
            logging.debug(
                f"grouped_llm_outputs {len(grouped_llm_outputs)} {dset_train.shape[0]}"
            )
            # ungroup responses
            llm_outputs = [None] * dset_train.shape[0]
            for group_obs_idxs, grouped_output in zip(
                obs_group_idxs, grouped_llm_outputs
            ):
                if grouped_output is not None:
                    for j, extraction in enumerate(
                        grouped_output.all_extractions[: len(group_obs_idxs)]
                    ):
                        llm_outputs[group_obs_idxs[j]] = extraction
                else:
                    logging.info(
                        f"warning: llm output was missing for idxs {group_obs_idxs}"
                    )
            logging.debug(f"llm_outputs {len(llm_outputs)} {dset_train.shape[0]}")
        else:
            if is_image:
                group_ids = np.arange(dset_train.shape[0])
                dataset = ImageDataset(dset_train.image_path.tolist(), prompt_template)
            else:
                group_ids, sentences = split_sentences_by_id(
                    dset_train, max_section_length, sentence_column
                )
                prompts = [prompt_template.replace("{sentence}", s) for s in sentences]
                dataset = TextDataset(
                    prompts,
                )

            loop = asyncio.get_running_loop()
            loop.set_debug(True)
            # Use await instead of asyncio.run()
            llm_outputs = await llm.get_outputs(
                dataset,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                is_image=is_image,
                temperature=0,
                max_retries=max_retries,
                response_model=ExtractResponseList,
            )

        # extract responses
        extracted_llm_outputs = _collate_extractions_by_group(
            llm_outputs, group_ids, len(batch_concepts_to_extract)
        )

        # fill in dictionary with extractions
        for idx, concept in enumerate(batch_concepts_to_extract):
            extracted_features = extracted_llm_outputs[:, idx : idx + 1]
            logging.info(
                "concept %s, prevalence %f", concept, np.mean(extracted_features)
            )
            all_extracted_features_dict[concept] = extracted_features

        if extraction_file is not None:
            with open(extraction_file, "wb") as f:
                pickle.dump(all_extracted_features_dict, f)

    return all_extracted_features_dict
