/**
 * Standalone Concept Analysis Interface
 * Client-side JavaScript for viewing clinical data with concepts from multiple initializations
 */

class StandaloneConceptAnalysisInterface {
    constructor() {
        // Data storage
        this.data = [];
        this.config = null;
        this.filteredData = [];
        
        // UI state
        this.currentPage = 1;
        this.perPage = 20;  // Reduced from 50 to prevent overflow
        this.currentSearch = '';
        this.partitionFilter = 'all';
        this.conceptFilter = '';
        this.selectedObservation = null;
        this.annotations = [];
        this.annotationIdCounter = 0;
        
        // File loading state
        this.dataLoaded = false;
        this.configLoaded = false;

        // Prompt editing state
        this.modifiedPrompts = {};
        this.editingPrompts = {};
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.showLoadingScreen();
    }
    
    showLoadingScreen() {
        const content = `
            <div class="text-center p-5">
                <h2>Clinical Data Viewer</h2>
                <p class="mb-4">Please load your data files to begin</p>
                <div class="row justify-content-center">
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-body">
                                <h5>Step 1: Load Configuration</h5>
                                <input type="file" id="configFileInput" class="form-control" accept=".json">
                                <small class="text-muted">Select config.json</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-body">
                                <h5>Step 2: Load Data</h5>
                                <input type="file" id="dataFileInput" class="form-control" accept=".csv" disabled>
                                <small class="text-muted">Select data.csv</small>
                            </div>
                        </div>
                    </div>
                </div>
                <div id="loadingStatus" class="mt-4"></div>
            </div>
        `;
        
        document.getElementById('mainContent').innerHTML = content;
        
        // Bind file input events
        document.getElementById('configFileInput').addEventListener('change', (e) => this.loadConfigFile(e));
        document.getElementById('dataFileInput').addEventListener('change', (e) => this.loadDataFile(e));
    }
    
    async loadConfigFile(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const text = await file.text();
            this.config = JSON.parse(text);
            this.configLoaded = true;
            
            document.getElementById('loadingStatus').innerHTML = 
                '<div class="alert alert-success">Configuration loaded successfully</div>';
            
            // Enable data file input
            document.getElementById('dataFileInput').disabled = false;
            
            // Initialize UI elements based on config
            this.initializeUIFromConfig();
            
        } catch (error) {
            document.getElementById('loadingStatus').innerHTML = 
                `<div class="alert alert-danger">Error loading config: ${error.message}</div>`;
        }
    }
    
    async loadDataFile(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        document.getElementById('loadingStatus').innerHTML = 
            '<div class="alert alert-info">Loading data... This may take a moment for large files.</div>';
        
        // Use Papa Parse for CSV parsing
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                this.data = results.data;
                this.filteredData = [...this.data];
                this.dataLoaded = true;
                
                document.getElementById('loadingStatus').innerHTML = 
                    `<div class="alert alert-success">Data loaded successfully (${this.data.length} rows)</div>`;
                
                // Initialize the main interface
                setTimeout(() => {
                    this.initializeMainInterface();
                }, 1000);
            },
            error: (error) => {
                document.getElementById('loadingStatus').innerHTML = 
                    `<div class="alert alert-danger">Error loading data: ${error.message}</div>`;
            }
        });
    }
    
    initializeUIFromConfig() {
        if (!this.config) return;
        
        // Update page title
        document.title = this.config.ui.title || 'Clinical Data Viewer';
        
        // Store config for later use
        this.idColumn = this.config.dataset.id_column;
        this.noteColumn = this.config.dataset.note_column;
        this.summaryColumn = this.config.dataset.summary_column;
        this.metadataColumns = this.config.dataset.metadata_columns || [];
        this.initializations = this.config.initializations || [];
        this.methodInfo = this.config.method_info || {};
    }
    
    initializeMainInterface() {
        // Load the main interface HTML
        this.loadMainInterfaceHTML();

        // Bind all events
        this.bindMainInterfaceEvents();

        // Load initial data
        this.updateConceptFilters();
        this.loadObservations();

        // Update health status
        this.updateHealthStatus();

        // Switch to Overview tab by default
        setTimeout(() => {
            const overviewTab = document.querySelector('[data-bs-target="#overview-tab"]');
            if (overviewTab) {
                overviewTab.click();
            }
        }, 100);
    }
    
    loadMainInterfaceHTML() {
        // This will be injected from the template
        const mainHTML = document.getElementById('mainInterfaceTemplate').innerHTML;
        document.getElementById('mainContent').innerHTML = mainHTML;
        
        // Update UI elements based on config
        document.getElementById('appTitle').textContent = this.config.ui.title;
        document.getElementById('observationsHeader').textContent = 
            this.config.ui.encounter_display_name || 'Observations';
        
        // Partition filter is always shown since we require partitions
        
        if (this.config.features.show_concepts && this.initializations.length > 0) {
            document.getElementById('conceptFilterContainer').style.display = 'block';
            document.getElementById('assignedConceptsSection').style.display = 'block';
            this.createInitializationTabs();
        }
        
        if (!this.config.features.show_summaries) {
            document.getElementById('summarySection').style.display = 'none';
        }
    }
    
    createInitializationTabs() {
        // Create tabs for each initialization in the assigned concepts section
        let tabsHTML = '<ul class="nav nav-tabs mb-3" role="tablist">';
        let tabContent = '<div class="tab-content">';
        
        this.initializations.forEach((init, index) => {
            const isActive = index === 0 ? 'active' : '';
            tabsHTML += `
                <li class="nav-item" role="presentation">
                    <button class="nav-link ${isActive}" data-bs-toggle="tab" 
                            data-bs-target="#init-tab-${init.id}" type="button">
                        ${init.name}
                    </button>
                </li>
            `;
            
            tabContent += `
                <div class="tab-pane fade ${isActive ? 'show active' : ''}" 
                     id="init-tab-${init.id}" role="tabpanel">
                    <div id="conceptsList-${init.id}" class="concepts-list">
                        <!-- Concepts will be populated here -->
                    </div>
                </div>
            `;
        });
        
        tabsHTML += '</ul>';
        tabContent += '</div>';
        
        document.getElementById('assignedConceptsList').innerHTML = tabsHTML + tabContent;
    }
    
    bindEvents() {
        // Placeholder for initial events - main events will be bound after interface loads
    }
    
    bindMainInterfaceEvents() {
        // Search input
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('input', debounce(() => {
                this.currentSearch = searchInput.value;
                this.currentPage = 1;
                this.loadObservations();
            }, 300));
        }
        
        // Partition filter
        const partitionFilter = document.getElementById('partitionFilter');
        if (partitionFilter) {
            partitionFilter.addEventListener('change', () => {
                this.partitionFilter = partitionFilter.value;
                this.currentPage = 1;
                this.loadObservations();
            });
        }
        
        // Concept filter
        const conceptFilter = document.getElementById('conceptFilter');
        if (conceptFilter) {
            conceptFilter.addEventListener('change', () => {
                this.conceptFilter = conceptFilter.value;
                this.currentPage = 1;
                this.loadObservations();
            });
        }
        
        // Export buttons
        document.getElementById('exportObservationsBtn')?.addEventListener('click', () => this.exportObservations());
        document.getElementById('exportSelectedEncounterBtn')?.addEventListener('click', () => this.exportSelectedEncounter());
        document.getElementById('clearAllAnnotationsBtn')?.addEventListener('click', () => this.clearAllAnnotations());
        
        // Text selection
        this.setupTextSelection();
        
        // Overview tab functionality
        this.bindOverviewTabEvents();
    }
    
    bindOverviewTabEvents() {
        // Handle tab switching
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (event) => {
                if (event.target.getAttribute('data-bs-target') === '#overview-tab') {
                    this.loadOverviewTab();
                }
            });
        });
    }
    
    loadOverviewTab() {
        // Load overview content
        this.renderMethodOverview();

        // Load enhanced prompts section
        const promptsContainer = document.getElementById('promptsContent');
        if (promptsContainer) {
            promptsContainer.innerHTML = this.renderPromptsSection();
            this.bindPromptCollapseEvents();
        }
    }
    
    renderMethodOverview() {
        const container = document.getElementById('methodOverviewContent');
        if (!container) return;
  
        const html = `
            <div class="mb-4 text-center">
                <h3 style="font-weight:700;color:#1f2937;">Why we need an LLM-aided Concept Bottleneck workflow</h3>
                <p class="lead text-muted mb-0">This page is the complete story: motivation, co-design process, today’s goals, and the interactive demo.</p>
            </div>
  
            <div class="presentation-section">
                <div class="section-header d-flex align-items-center" data-bs-toggle="collapse" data-bs-target="#overview-classical" aria-expanded="true">
                    <h4 class="mb-0"><i class="fas fa-heartbeat me-2"></i>Classical risk prediction models</h4>
                    <i class="fas fa-chevron-down ms-auto"></i>
                </div>
                <div id="overview-classical" class="section-content collapse show">
                    <div class="row g-4">
                        <div class="col-lg-6">
                            <div class="card h-100 shadow-sm border-0" style="background-color:#e8f4fd;">
                                <div class="card-body">
                                    <h5 class="text-primary"><i class="fas fa-user-md me-2"></i>Clinician intuition drives features</h5>
                                    <p>"Let’s build a model to predict hospital readmission risk. I think the most relevant features are…"</p>
                                    <ul class="mb-0">
                                        <li>Number of prior hospitalizations</li>
                                        <li>Hemoglobin at discharge</li>
                                        <li>Smoking status</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-lg-6">
                            <div class="card h-100 shadow-sm border-0" style="background-color:#fff4d6;">
                                <div class="card-body">
                                    <h5 class="text-warning"><i class="fas fa-chart-bar me-2"></i>What the analyst returns</h5>
                                    <p class="mb-3">Great! Here is the fitted logistic regression model using those signals.</p>
                                    <table class="table table-bordered mb-3">
                                        <thead class="table-warning">
                                            <tr><th>Feature</th><th>Coefficient</th></tr>
                                        </thead>
                                        <tbody>
                                            <tr><td>(Intercept)</td><td>-0.3</td></tr>
                                            <tr><td>Hospitalizations</td><td>1.5</td></tr>
                                            <tr><td>Hemoglobin</td><td>-0.5</td></tr>
                                            <tr><td>Smoking</td><td>0.4</td></tr>
                                        </tbody>
                                    </table>
                                    <p class="mb-0"><strong>AUC 0.72.</strong> Helpful, but it hides the richer reasoning clinicians want to see.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
  
            <div class="presentation-section">
                <div class="section-header d-flex align-items-center" data-bs-toggle="collapse" data-bs-target="#overview-codesign" aria-expanded="true">
                    <h4 class="mb-0"><i class="fas fa-people-arrows me-2"></i>Idealized human-only co-design</h4>
                    <i class="fas fa-chevron-down ms-auto"></i>
                </div>
                <div id="overview-codesign" class="section-content collapse show">
                    <p class="text-muted mb-4">Real-world cycles take weeks because every iteration requires a new build, validation, and interpretation pass.</p>
                    <div class="flow-steps">
                        <div class="flow-step">
                            <div class="step-number">1</div>
                            <div class="step-content">
                                <h6>Clinician proposes features</h6>
                                <p class="mb-2">“To predict readmission risk, I think the most relevant features are…”</p>
                                <span class="badge bg-warning text-dark">≈ 2 weeks</span>
                            </div>
                        </div>
                        <div class="flow-arrow"><i class="fas fa-chevron-right"></i></div>
                        <div class="flow-step">
                            <div class="step-number">2</div>
                            <div class="step-content">
                                <h6>Data scientist inspects model</h6>
                                <p class="mb-2">“The model depends on … and achieves an AUC of …”</p>
                                <span class="badge bg-warning text-dark">≈ 3 weeks</span>
                            </div>
                        </div>
                        <div class="flow-arrow"><i class="fas fa-chevron-right"></i></div>
                        <div class="flow-step">
                            <div class="step-number">3</div>
                            <div class="step-content">
                                <h6>Clinical review & revision</h6>
                                <p class="mb-2">“The reasoning doesn’t look right. Let’s modify the features…”</p>
                                <span class="badge bg-warning text-dark">≈ 2 weeks</span>
                            </div>
                        </div>
                    </div>
                    <p class="mt-3 mb-0 text-center text-danger fw-bold">Weeks of iteration → slow learning and hard-to-scale collaboration.</p>
                </div>
            </div>
  
            <div class="presentation-section">
                <div class="section-header d-flex align-items-center" data-bs-toggle="collapse" data-bs-target="#overview-data" aria-expanded="true">
                    <h4 class="mb-0"><i class="fas fa-balance-scale me-2"></i>Combining the best of both worlds</h4>
                    <i class="fas fa-chevron-down ms-auto"></i>
                </div>
                <div id="overview-data" class="section-content collapse show">
                    <div class="row g-4 align-items-center">
                        <div class="col-md-6">
                            <div class="card shadow-sm">
                                <div class="card-body">
                                    <h5><i class="fas fa-table me-2"></i>Traditional tabular features</h5>
                                    <ul class="mb-0">
                                        <li>Age</li>
                                        <li># prior hospitalizations</li>
                                        <li>Creatinine level</li>
                                        <li>On ventilator?</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card shadow-sm">
                                <div class="card-body">
                                    <h5><i class="fas fa-notes-medical me-2"></i>Concepts from clinical notes</h5>
                                    <ul class="mb-0">
                                        <li>Family requesting a specific facility?</li>
                                        <li>Awaiting insurance authorization?</li>
                                        <li>Requiring IV antibiotics at discharge?</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                    <p class="mt-4 mb-0 text-center text-muted">The concept bottleneck gives clinicians an interpretable language that spans both data modalities.</p>
                </div>
            </div>
  
            <div class="presentation-section">
                <div class="section-header d-flex align-items-center" data-bs-toggle="collapse" data-bs-target="#overview-llm" aria-expanded="true">
                    <h4 class="mb-0"><i class="fas fa-robot me-2"></i>LLM-aided co-design loop</h4>
                    <i class="fas fa-chevron-down ms-auto"></i>
                </div>
                <div id="overview-llm" class="section-content collapse show">
                    <div class="row gy-3">
                        <div class="col-12">
                            <div class="p-3 rounded" style="background-color:#e8f4fd;">
                                <strong><i class="fas fa-user-md text-primary me-2"></i>Clinician:</strong> Goal: predict hospital readmission risk using clinical notes + tabular data.
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="p-3 rounded" style="background-color:#fff3cd;">
                                <strong><i class="fas fa-people-carry text-warning me-2"></i>Data team:</strong> Fit a compact logistic regression on 5 interpretable concepts.
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="p-3 rounded" style="background-color:#e9f5ee;">
                                <strong><i class="fas fa-robot text-success me-2"></i>LLM:</strong> Suggest candidate concepts to extract, highlight coverage, and surface ambiguities.
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="p-3 rounded" style="background-color:#dbeafe;">
                                <strong><i class="fas fa-chart-line text-primary me-2"></i>Model evaluation:</strong> Train, score, and report performance + example extractions.
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="p-3 rounded" style="background-color:#f0f9ff;">
                                <strong><i class="fas fa-sync-alt text-info me-2"></i>Iterate:</strong> Swap concepts, adjust wording, and re-run—minutes instead of weeks.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
  
            <div class="presentation-section">
                <div class="section-header d-flex align-items-center" data-bs-toggle="collapse" data-bs-target="#overview-meeting" aria-expanded="true">
                    <h4 class="mb-0"><i class="fas fa-handshake me-2"></i>Today’s co-design session</h4>
                    <i class="fas fa-chevron-down ms-auto"></i>
                </div>
                <div id="overview-meeting" class="section-content collapse show">
                    <div class="alert alert-info mb-4">
                        <h5 class="mb-3"><i class="fas fa-bullseye me-2"></i>Why we’re meeting</h5>
                        <ul class="mb-0">
                            <li>Share motivating patient stories and summaries generated by the LLM.</li>
                            <li>Review the initial concepts surfaced by the model and spot gaps or redundancies.</li>
                            <li>Capture your feedback to refine prompts and guide the next training cycle.</li>
                        </ul>
                    </div>
                    <p class="text-muted mb-0 text-center">Everything below (summaries, concept lists, prompts) updates live as we incorporate your suggestions.</p>
                </div>
            </div>
  
            <div class="presentation-section">
                <div class="section-header d-flex align-items-center" data-bs-toggle="collapse" data-bs-target="#overview-pipeline" aria-expanded="true">
                    <h4 class="mb-0"><i class="fas fa-project-diagram me-2"></i>LLM-guided concept bottleneck pipeline</h4>
                    <i class="fas fa-chevron-down ms-auto"></i>
                </div>
                <div id="overview-pipeline" class="section-content collapse show">
                    <div class="enhanced-flow-diagram">
                        <h3>How the method works end-to-end</h3>
                        <div class="flow-steps justify-content-center">
                            <div class="flow-step-enhanced has-feedback">
                                <div class="feedback-badge">★</div>
                                <div class="step-number">1</div>
                                <div class="step-content">
                                    <h6>Extract keyphrases</h6>
                                    <p>Surface salient events from notes (e.g., chest pain, insurance delays).</p>
                                </div>
                            </div>
                            <div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div>
                            <div class="flow-step-enhanced has-feedback">
                                <div class="feedback-badge">★</div>
                                <div class="step-number">2</div>
                                <div class="step-content">
                                    <h6>Propose candidate concepts</h6>
                                    <p>LLM turns keyphrases into yes/no questions with priors.</p>
                                </div>
                            </div>
                            <div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div>
                            <div class="flow-step-enhanced">
                                <div class="step-number">3</div>
                                <div class="step-content">
                                    <h6>Extract concept values</h6>
                                    <p>Score every encounter on each candidate concept.</p>
                                </div>
                            </div>
                            <div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div>
                            <div class="flow-step-enhanced has-feedback">
                                <div class="feedback-badge">★</div>
                                <div class="step-number">4</div>
                                <div class="step-content">
                                    <h6>Select & evaluate</h6>
                                    <p>Choose predictive concepts, update prompts, and repeat with clinician feedback.</p>
                                </div>
                            </div>
                        </div>
                        <p class="text-muted mt-3 mb-0">Stars mark the touchpoints where subject-matter expertise is most valuable.</p>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = html;
        this.updatePresentationProgress();
        this.loadOverviewSections();
    }

    renderFeedbackOpportunities() {
        const container = document.getElementById('feedbackOpportunitiesContent');
        if (!container) return;

        const html = `
            <div class="feedback-opportunities">
                <h3 style="text-align: center; margin-bottom: 2rem; color: #2c3e50;">Areas for Clinical Feedback</h3>

                <!-- Feedback Point 1: Summary Generation -->
                <div class="feedback-section">
                    <div class="feedback-icon">1</div>
                    <h5><i class="fas fa-clipboard-list me-2"></i>Hospital Course Summary Generation</h5>

                    <div class="feedback-what">
                        <strong><i class="fas fa-question-circle me-2"></i>What feedback do we need?</strong>
                        <p>Help us identify critical events, processes, and factors during hospitalization that significantly impact length of stay but might be missed by our current approach.</p>
                    </div>

                    <div class="feedback-how">
                        <strong><i class="fas fa-tools me-2"></i>How does this help?</strong>
                        <p>Your input shapes the foundation of our entire method by ensuring summaries capture the most relevant clinical information for length of stay prediction.</p>
                    </div>

                    <div class="feedback-example">
                        <strong><i class="fas fa-lightbulb me-2"></i>Example questions for you:</strong>
                        <ul>
                            <li>Are we missing important coordination delays (e.g., waiting for insurance approval, family meetings)?</li>
                            <li>Should we emphasize certain types of complications more heavily?</li>
                            <li>Are there specific discharge planning factors we're not capturing?</li>
                        </ul>
                    </div>
                </div>

                <!-- Feedback Point 2: Iterative Refinement -->
                <div class="feedback-section">
                    <div class="feedback-icon">2</div>
                    <h5><i class="fas fa-sync-alt me-2"></i>Iterative Concept Generation & Refinement</h5>

                    <div class="feedback-what">
                        <strong><i class="fas fa-question-circle me-2"></i>What feedback do we need?</strong>
                        <p>Help us identify missing concepts, redundant concepts, and opportunities to make concepts more clinically specific and actionable.</p>
                    </div>

                    <div class="feedback-how">
                        <strong><i class="fas fa-tools me-2"></i>How does this help?</strong>
                        <p>Improves both model performance and clinical interpretability by ensuring our concept set is comprehensive yet focused on actionable factors.</p>
                    </div>

                    <div class="feedback-example">
                        <strong><i class="fas fa-lightbulb me-2"></i>Example feedback areas:</strong>
                        <ul>
                            <li>Are there important concepts we're missing entirely?</li>
                            <li>Can similar concepts be consolidated for clarity?</li>
                            <li>Should certain concepts be split into more specific variants?</li>
                            <li>Are any concepts too vague to be clinically useful?</li>
                        </ul>
                    </div>
                </div>

            </div>
        `;

        container.innerHTML = html;
    }

    renderConceptExamples() {
        const container = document.getElementById('conceptExamplesContent');
        if (!container) return;

        let html = `
            <div class="concept-examples">
                <h3 style="text-align: center; margin-bottom: 2rem; color: #2c3e50;">Real Concept Examples from Sepsis Length of Stay Study</h3>

                <p style="font-size: 1.2rem; text-align: center; margin-bottom: 3rem; color: #495057;">
                    Here are actual binary concepts extracted by our method, showing how clinical factors become yes/no questions:
                </p>
        `;

        // The binary concepts have been moved to their own section

        container.innerHTML = html;
    }

    renderExampleSummaries() {
        const container = document.getElementById('exampleSummariesContent');
        if (!container) return;

        let html = '';

        // Check if example summaries are available
        if (window.exampleSummaries && window.exampleSummaries.length > 0) {
            html += `
                <div class="example-summaries">
                    <h4 style="text-align: center; margin-bottom: 2rem; color: #2c3e50;"><i class="fas fa-file-medical me-2"></i>Example Clinical Summaries</h4>
                    <div class="summaries-carousel">
            `;

            window.exampleSummaries.forEach((example, index) => {
                const summaryContent = typeof marked !== 'undefined' ?
                    marked.parse(example.summary || 'Summary not available') :
                    this.escapeHtml(example.summary || 'Summary not available');

                html += `
                    <div class="summary-example mb-4" style="display: ${index === 0 ? 'block' : 'none'};" data-index="${index}">
                        <div class="col-12">
                            <div class="card">
                                <div class="card-body">
                                    <div class="summary-content" style="font-size: 1rem; line-height: 1.6;">${summaryContent}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            // Add navigation controls if more than one example
            if (window.exampleSummaries.length > 1) {
                html += `
                    <div class="carousel-controls text-center mt-3">
                        <button class="btn btn-outline-primary me-2" onclick="viewer.previousSummaryExample()">
                            <i class="fas fa-chevron-left"></i> Previous
                        </button>
                        <span class="mx-3" id="summaryCounter">1 of ${window.exampleSummaries.length}</span>
                        <button class="btn btn-outline-primary ms-2" onclick="viewer.nextSummaryExample()">
                            Next <i class="fas fa-chevron-right"></i>
                        </button>
                    </div>
                `;
            }

            html += `
                    </div>
                </div>
            `;
        } else {
            html = `
                <div class="text-center text-muted">
                    <p>Example summaries will be displayed here when available.</p>
                </div>
            `;
        }

        container.innerHTML = html;
        this.currentSummaryIndex = 0;
    }

    nextSummaryExample() {
        if (!window.exampleSummaries || window.exampleSummaries.length <= 1) return;

        const current = document.querySelector(`[data-index="${this.currentSummaryIndex}"]`);
        current.style.display = 'none';

        this.currentSummaryIndex = (this.currentSummaryIndex + 1) % window.exampleSummaries.length;

        const next = document.querySelector(`[data-index="${this.currentSummaryIndex}"]`);
        next.style.display = 'block';

        document.getElementById('summaryCounter').textContent = `${this.currentSummaryIndex + 1} of ${window.exampleSummaries.length}`;
    }

    previousSummaryExample() {
        if (!window.exampleSummaries || window.exampleSummaries.length <= 1) return;

        const current = document.querySelector(`[data-index="${this.currentSummaryIndex}"]`);
        current.style.display = 'none';

        this.currentSummaryIndex = this.currentSummaryIndex === 0 ? window.exampleSummaries.length - 1 : this.currentSummaryIndex - 1;

        const next = document.querySelector(`[data-index="${this.currentSummaryIndex}"]`);
        next.style.display = 'block';

        document.getElementById('summaryCounter').textContent = `${this.currentSummaryIndex + 1} of ${window.exampleSummaries.length}`;
    }

    renderGeneratedConcepts() {
        const container = document.getElementById('generatedConceptsContent');
        if (!container) return;

        let html = '';

        // Add binary concepts display
        if (this.initializations.length > 0) {
            html += `
                <div class="training-summary">
                    <h4 style="text-align: center; margin-bottom: 2rem; color: #2c3e50;"><i class="fas fa-list-check me-2"></i>Generated Binary Concepts</h4>
                    <p class="text-muted mb-4" style="text-align: center; font-size: 1.1rem;">These are the actual concepts generated by our method for sepsis length of stay prediction:</p>
                    <div class="row">
            `;

            const init = this.initializations[0];
            const concepts = init.concepts || [];
            html += `
                <div class="col-md-12">
                    <div class="card h-100" style="border: 1px solid #0d6efd;">
                        <div class="card-body">
                            <div class="concept-list">
                                ${concepts.length > 0 ? `
                                    <ul class="list-unstyled" style="font-size: 1.1rem; line-height: 1.6; max-height: 400px; overflow-y: auto;">
                                        ${concepts.map(concept => `
                                            <li class="mb-2 p-2" style="background-color: #f8f9fa; border-radius: 0.25rem; border-left: 3px solid #0d6efd;">
                                                <i class="fas fa-question-circle text-primary me-2"></i>
                                                ${concept}
                                            </li>
                                        `).join('')}
                                    </ul>
                                ` : '<div class="text-muted text-center">No concepts available</div>'}
                            </div>
                        </div>
                    </div>
                </div>
            `;

            html += `
                    </div>
                </div>
            `;
        } else {
            html = `
                <div class="text-center text-muted">
                    <p>Generated concepts will be displayed here when data is loaded.</p>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    loadOverviewSections() {
        // Load example summaries
        this.renderExampleSummaries();

        // Load generated concepts
        this.renderGeneratedConcepts();

        // Load feedback opportunities
        this.renderFeedbackOpportunities();

        // Load concept examples
        // this.renderConceptExamples();
    }

    // Presentation mode functions
    togglePresentationMode() {
        document.body.classList.toggle('presentation-mode');
    }

    previousSection() {
        // Simple scroll to previous section
        const sections = document.querySelectorAll('#overview-tab .card');
        const currentScroll = window.scrollY;
        let targetSection = sections[0];

        for (let i = sections.length - 1; i >= 0; i--) {
            if (sections[i].offsetTop < currentScroll - 100) {
                targetSection = sections[i];
                break;
            }
        }

        targetSection.scrollIntoView({ behavior: 'smooth' });
        this.updatePresentationProgress();
    }

    nextSection() {
        // Simple scroll to next section
        const sections = document.querySelectorAll('#overview-tab .card');
        const currentScroll = window.scrollY;
        let targetSection = sections[sections.length - 1];

        for (let i = 0; i < sections.length; i++) {
            if (sections[i].offsetTop > currentScroll + 100) {
                targetSection = sections[i];
                break;
            }
        }

        targetSection.scrollIntoView({ behavior: 'smooth' });
        this.updatePresentationProgress();
    }

    updatePresentationProgress() {
        const sections = document.querySelectorAll('#overview-tab .card');
        const currentScroll = window.scrollY;
        let currentSection = 1;

        for (let i = 0; i < sections.length; i++) {
            if (sections[i].offsetTop <= currentScroll + 200) {
                currentSection = i + 1;
            }
        }

        const progressElement = document.getElementById('presentationProgress');
        if (progressElement) {
            progressElement.textContent = `${currentSection}/${sections.length}`;
        }
    }

    bindPromptCollapseEvents() {
        // Handle chevron rotation for prompt collapse sections
        document.querySelectorAll('[data-bs-toggle="collapse"]').forEach(button => {
            const targetId = button.getAttribute('data-bs-target');
            const target = document.querySelector(targetId);
            const icon = button.querySelector('i.fas');
            
            if (target && icon) {
                target.addEventListener('show.bs.collapse', () => {
                    icon.classList.remove('fa-chevron-right');
                    icon.classList.add('fa-chevron-down');
                });
                
                target.addEventListener('hide.bs.collapse', () => {
                    icon.classList.remove('fa-chevron-down');
                    icon.classList.add('fa-chevron-right');
                });
            }
        });
    }
    // Helper function to wrap text into multiple lines
    wrapText(text, maxWidth) {
        const words = text.split(/\s+/);
        const lines = [];
        let currentLine = '';
        
        words.forEach(word => {
            const testLine = currentLine ? `${currentLine} ${word}` : word;
            // Approximate character width (assuming ~6px per character)
            if (testLine.length * 6 > maxWidth) {
                if (currentLine) {
                    lines.push(currentLine);
                    currentLine = word;
                } else {
                    // Word is too long, force break
                    lines.push(word);
                }
            } else {
                currentLine = testLine;
            }
        });
        
        if (currentLine) {
            lines.push(currentLine);
        }
        
        return lines;
    }
    
    buildHierarchy(linkage, concepts) {
        if (linkage.length === 0 || concepts.length === 1) {
            return { 
                name: concepts[0] ? this.normalizeConceptText(concepts[0].text) : 'Unknown',
                concept: concepts[0] 
            };
        }
        
        // Create a simpler hierarchy for small datasets
        if (concepts.length <= 3) {
            return {
                name: 'root',
                children: concepts.map(concept => ({
                    name: this.normalizeConceptText(concept.text),
                    concept: concept
                }))
            };
        }
        
        // For larger datasets, create a simple binary tree
        const leaves = concepts.map(concept => ({
            name: this.normalizeConceptText(concept.text),
            concept: concept
        }));
        
        // Group leaves into pairs and build tree bottom-up
        let currentLevel = [...leaves];
        
        while (currentLevel.length > 1) {
            const nextLevel = [];
            
            for (let i = 0; i < currentLevel.length; i += 2) {
                if (i + 1 < currentLevel.length) {
                    // Create internal node with two children
                    nextLevel.push({
                        name: `cluster_${nextLevel.length}`,
                        children: [currentLevel[i], currentLevel[i + 1]]
                    });
                } else {
                    // Odd number, carry forward the last node
                    nextLevel.push(currentLevel[i]);
                }
            }
            
            currentLevel = nextLevel;
        }
        
        return currentLevel[0];
    }
    
    // Normalize concept text for display
    normalizeConceptText(text) {
        return text
            .replace(/^Does the note mention (that )?/, '')
            .replace(/^Does /, '')
            .replace(/\?$/, '')
            .trim();
    }
    
    
    renderPromptsSection() {
        if (!this.methodInfo.prompts) return '';

        let html = `
            <div class="mt-4">
                <h3 style="text-align: center; margin-bottom: 2rem; color: #2c3e50;">Prompts & Feedback Integration Points</h3>
                <div class="alert alert-info mb-4">
                    <h5><i class="fas fa-info-circle me-2"></i>How Feedback Gets Integrated</h5>
                    <p>Methods for modifying these prompts:</p>
                    <ul class="mb-0">
                        <li><strong>Examples:</strong> We add examples of good vs. poor concept phrasing</li>
                        <li><strong>Focus Areas:</strong> We emphasize clinical factors you identify as important</li>
                        <li><strong>Terminology:</strong> We adjust language to match clinical conventions</li>
                        <li><strong>Context:</strong> We provide better context about length of stay relationships</li>
                    </ul>
                </div>

                <div class="text-center mb-4">
                    <button class="btn btn-success" onclick="app.exportModifiedPrompts()" id="exportPromptsBtn" ${Object.keys(this.modifiedPrompts).length === 0 ? 'disabled' : ''}>
                        <i class="fas fa-download me-2"></i>Export Modified Prompts (${Object.keys(this.modifiedPrompts).length})
                    </button>
                </div>
        `;

        const promptCategories = {
            'step1_summary': {
                title: 'Step 1: Hospital Course Summary Generation',
                feedback: 'dentify which clinical events and processes most impact length of stay',
                icon: 'fas fa-clipboard-list'
            },
            'step4_generation': {
                title: 'Step 5: Iterative Concept Generation & Refinement',
                feedback: 'Identify missing concepts and improve existing ones',
                icon: 'fas fa-sync-alt'
            }
        };

        // Display prompts in the correct order matching CBM training steps
        const orderedSteps = ['step1_summary',  'step4_generation'];

        orderedSteps.forEach(category => {
            const prompts = this.methodInfo.prompts[category];
            if (!prompts || prompts.length === 0) return;

            const categoryInfo = promptCategories[category] || { title: category, feedback: '', icon: 'fas fa-cog' };

            html += `
                <div class="prompt-category mb-4">
                    <div class="d-flex align-items-center mb-3">
                        <h6 class="text-primary border-bottom pb-2 flex-grow-1">
                            <i class="${categoryInfo.icon} me-2"></i>${categoryInfo.title}
                        </h6>
                    </div>
            `;

            prompts.forEach((prompt, index) => {
                const collapseId = `collapse-${category}-${index}`;
                const promptId = `${category}-${index}`;
                const isModified = this.modifiedPrompts[promptId];
                const isEditing = this.editingPrompts[promptId];

                html += `
                    <div class="card mb-2">
                        <div class="card-header bg-light">
                            <div class="d-flex justify-content-between align-items-center">
                                <button class="btn btn-link btn-sm text-decoration-none fw-bold p-0" type="button"
                                        data-bs-toggle="collapse" data-bs-target="#${collapseId}"
                                        aria-expanded="false" aria-controls="${collapseId}">
                                    <i class="fas fa-chevron-right me-2" id="icon-${collapseId}"></i>
                                    ${prompt.filename}
                                    ${isModified ? '<i class="fas fa-circle ms-2 text-success" title="Modified" style="font-size: 0.5rem;"></i>' : ''}
                                </button>
                                <div class="btn-group btn-group-sm">
                                    <button class="btn btn-outline-primary btn-sm" onclick="app.togglePromptEdit('${promptId}')" id="edit-btn-${promptId}">
                                        <i class="fas fa-${isEditing ? 'save' : 'edit'}"></i> ${isEditing ? 'Save' : 'Edit'}
                                    </button>
                                    ${isModified ? `<button class="btn btn-outline-secondary btn-sm" onclick="app.resetPrompt('${promptId}')" title="Reset to original"><i class="fas fa-undo"></i></button>` : ''}
                                </div>
                            </div>
                        </div>
                        <div id="${collapseId}" class="collapse">
                            <div class="card-body p-3">
                                <div class="prompt-text-container">
                                    <div id="prompt-view-${promptId}" style="display: ${isEditing ? 'none' : 'block'}">
                                        <pre class="prompt-text-full">${this.escapeHtml(isModified ? this.modifiedPrompts[promptId] : prompt.content)}</pre>
                                    </div>
                                    <div id="prompt-edit-${promptId}" style="display: ${isEditing ? 'block' : 'none'}">
                                        <textarea class="form-control" rows="20" style="font-family: monospace;" id="prompt-textarea-${promptId}">${this.escapeHtml(isModified ? this.modifiedPrompts[promptId] : prompt.content)}</textarea>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += `</div>`;
        });

        html += `
            </div>
        `;
        return html;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    togglePromptEdit(promptId) {
        const isEditing = this.editingPrompts[promptId];

        if (isEditing) {
            // Save the changes
            const textarea = document.getElementById(`prompt-textarea-${promptId}`);
            const originalPrompt = this.getOriginalPrompt(promptId);

            if (textarea.value !== originalPrompt) {
                this.modifiedPrompts[promptId] = textarea.value;
            } else {
                delete this.modifiedPrompts[promptId];
            }

            this.editingPrompts[promptId] = false;

            // Update the view
            document.getElementById(`prompt-view-${promptId}`).style.display = 'block';
            document.getElementById(`prompt-edit-${promptId}`).style.display = 'none';

            // Update the pre content
            const preElement = document.querySelector(`#prompt-view-${promptId} pre`);
            preElement.textContent = this.modifiedPrompts[promptId] || originalPrompt;

        } else {
            // Enter edit mode
            this.editingPrompts[promptId] = true;

            document.getElementById(`prompt-view-${promptId}`).style.display = 'none';
            document.getElementById(`prompt-edit-${promptId}`).style.display = 'block';
        }

        // Update the edit button text without re-rendering
        const editButton = document.querySelector(`[onclick="viewer.togglePromptEdit('${promptId}')"]`);
        if (editButton) {
            const isEditing = this.editingPrompts[promptId];
            const isModified = this.modifiedPrompts[promptId] !== undefined;
            const modifiedBadge = isModified ? ' <span class="badge bg-warning text-dark ms-1">Modified</span>' : '';
            editButton.innerHTML = `<i class="fas fa-${isEditing ? 'save' : 'edit'}"></i> ${isEditing ? 'Save' : 'Edit'}${isEditing ? '' : modifiedBadge}`;
        }
    }

    resetPrompt(promptId) {
        if (confirm('Are you sure you want to reset this prompt to its original content?')) {
            delete this.modifiedPrompts[promptId];
            this.editingPrompts[promptId] = false;

            // Re-render to update display
            const promptsContainer = document.getElementById('promptsContent');
            if (promptsContainer) {
                promptsContainer.innerHTML = this.renderPromptsSection();
                this.bindPromptCollapseEvents();
            }
        }
    }

    getOriginalPrompt(promptId) {
        const [category, index] = promptId.split('-');
        const prompts = this.methodInfo.prompts?.[category];
        if (prompts && prompts[parseInt(index)]) {
            return prompts[parseInt(index)].content;
        }
        return '';
    }

    exportModifiedPrompts() {
        if (Object.keys(this.modifiedPrompts).length === 0) {
            alert('No prompts have been modified yet.');
            return;
        }

        const exportData = {
            export_date: new Date().toISOString(),
            clinical_feedback_session: {
                total_modified_prompts: Object.keys(this.modifiedPrompts).length,
                modifications: {}
            }
        };

        // Organize modifications by category
        Object.keys(this.modifiedPrompts).forEach(promptId => {
            const [category, index] = promptId.split('-');
            const originalPrompt = this.getOriginalPrompt(promptId);
            const modifiedPrompt = this.modifiedPrompts[promptId];

            if (!exportData.clinical_feedback_session.modifications[category]) {
                exportData.clinical_feedback_session.modifications[category] = [];
            }

            const prompts = this.methodInfo.prompts?.[category];
            const filename = prompts && prompts[parseInt(index)] ? prompts[parseInt(index)].filename : `prompt_${index}`;

            exportData.clinical_feedback_session.modifications[category].push({
                filename: filename,
                prompt_index: parseInt(index),
                original_content: originalPrompt,
                modified_content: modifiedPrompt,
                change_summary: this.generateChangeSummary(originalPrompt, modifiedPrompt)
            });
        });

        const json = JSON.stringify(exportData, null, 2);
        const filename = `clinical_feedback_prompts_${new Date().toISOString().split('T')[0]}.json`;
        this.downloadFile(json, filename, 'application/json');
    }

    generateChangeSummary(original, modified) {
        const originalLines = original.split('\n').length;
        const modifiedLines = modified.split('\n').length;
        const lineDiff = modifiedLines - originalLines;

        return {
            original_lines: originalLines,
            modified_lines: modifiedLines,
            lines_added_removed: lineDiff,
            character_count_change: modified.length - original.length
        };
    }
    
    updateHealthStatus() {
        const status = {
            totalRows: this.data.length,
            conceptsAvailable: this.initializations.reduce((sum, init) => sum + init.concepts.length, 0),
            initializations: this.initializations.length
        };
        
        document.getElementById('healthStatus').innerHTML = `
            <div class="text-white">
                <small>
                    Loaded: ${status.totalRows} observations | 
                    ${status.initializations} initializations | 
                    ${status.conceptsAvailable} concepts
                </small>
            </div>
        `;
    }
    
    updateConceptFilters() {
        const conceptFilter = document.getElementById('conceptFilter');
        if (!conceptFilter || this.initializations.length === 0) return;
        
        let optionsHTML = '<option value="">All Concepts</option>';
        
        this.initializations.forEach(init => {
            optionsHTML += `<optgroup label="${init.name}">`;
            init.concepts.forEach((concept, index) => {
                const value = `${init.concept_prefix}_concept_${index}`;
                optionsHTML += `<option value="${value}">${concept}</option>`;
            });
            optionsHTML += '</optgroup>';
        });
        
        conceptFilter.innerHTML = optionsHTML;
    }
    
    loadObservations() {
        // Filter data
        this.filterData();
        
        // Calculate pagination
        const totalPages = Math.ceil(this.filteredData.length / this.perPage);
        const start = (this.currentPage - 1) * this.perPage;
        const end = start + this.perPage;
        const pageData = this.filteredData.slice(start, end);
        
        // Render observations list
        this.renderObservationsList(pageData);
        
        // Render pagination
        this.renderPagination(totalPages);
    }
    
    filterData() {
        this.filteredData = this.data.filter(row => {
            // Note: CSV is already filtered to only include train/test observations
            
            // Search filter
            if (this.currentSearch) {
                const searchLower = this.currentSearch.toLowerCase();
                const noteText = (row[this.noteColumn] || '').toLowerCase();
                const diagCode = (row.parent_diagnosis_code || '').toLowerCase();
                if (!noteText.includes(searchLower) && !diagCode.includes(searchLower)) {
                    return false;
                }
            }
            
            // Partition filter
            if (this.partitionFilter !== 'all') {
                if (this.partitionFilter === 'train' && row.partition !== 'train') return false;
                if (this.partitionFilter === 'test' && row.partition !== 'test') return false;
            }
            
            // Concept filter
            if (this.conceptFilter && row[this.conceptFilter] !== undefined) {
                if (row[this.conceptFilter] <= 0.5) return false;
            }
            
            return true;
        });
    }
    
    renderObservationsList(observations) {
        const container = document.getElementById('observationsList');
        if (!container) return;
        
        if (observations.length === 0) {
            container.innerHTML = '<div class="text-center p-3 text-muted">No observations found</div>';
            return;
        }
        
        let html = '';
        observations.forEach(obs => {
            const isSelected = this.selectedObservation && 
                              obs[this.idColumn] === this.selectedObservation[this.idColumn];
            
            html += `
                <div class="observation-row ${isSelected ? 'selected' : ''}" 
                     data-id="${obs[this.idColumn]}"
                     onclick="app.selectObservation('${obs[this.idColumn]}')">
                    <div class="fw-bold">${this.config.ui.encounter_display_name} ${obs[this.idColumn]}</div>
                    <div class="small">
                        ${this.renderObservationMetadata(obs)}
                    </div>
                    ${this.renderObservationBadges(obs)}
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    renderObservationMetadata(obs) {
        const displayColumns = ['parent_diagnosis_code', 'length_of_stay_days', 'global_median_los'];
        return displayColumns
            .map(col => {
                if (obs[col] !== undefined) {
                    const displayName = this.config.ui.column_display_names?.[col] || col;
                    let value = obs[col];
                    
                    // Round LOS values to 2 decimal places
                    if (col === 'length_of_stay_days' || col === 'global_median_los') {
                        value = parseFloat(value).toFixed(2);
                    }
                    
                    return `${displayName}: ${value}`;
                }
                return '';
            })
            .filter(s => s)
            .join(' | ');
    }
    
    renderObservationBadges(obs) {
        let badges = '';
        
        // Partition badge
        if (obs.partition) {
            const badgeClass = obs.partition === 'train' ? 'bg-primary' : 'bg-info';
            badges += `<span class="badge ${badgeClass} me-1">${obs.partition}</span>`;
        }
        
        // Concept count badges for each initialization
        this.initializations.forEach(init => {
            const assignedCol = `assigned_concepts_${init.concept_prefix}`;
            if (obs[assignedCol]) {
                const concepts = obs[assignedCol].split(',').filter(c => c.trim());
                if (concepts.length > 0) {
                    badges += `<span class="badge bg-secondary me-1">${init.name}: ${concepts.length}</span>`;
                }
            }
        });
        
        return badges ? `<div class="mt-1">${badges}</div>` : '';
    }
    
    renderPagination(totalPages) {
        const container = document.getElementById('pagination');
        if (!container || totalPages <= 1) {
            container.innerHTML = '';
            return;
        }
        
        let html = '<nav><ul class="pagination pagination-sm justify-content-center mb-0">';
        
        // Previous button
        html += `
            <li class="page-item ${this.currentPage === 1 ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="app.goToPage(${this.currentPage - 1})">Previous</a>
            </li>
        `;
        
        // Page numbers
        const maxPages = 5;
        let startPage = Math.max(1, this.currentPage - Math.floor(maxPages / 2));
        let endPage = Math.min(totalPages, startPage + maxPages - 1);
        
        if (endPage - startPage < maxPages - 1) {
            startPage = Math.max(1, endPage - maxPages + 1);
        }
        
        for (let i = startPage; i <= endPage; i++) {
            html += `
                <li class="page-item ${i === this.currentPage ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="app.goToPage(${i})">${i}</a>
                </li>
            `;
        }
        
        // Next button
        html += `
            <li class="page-item ${this.currentPage === totalPages ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="app.goToPage(${this.currentPage + 1})">Next</a>
            </li>
        `;
        
        html += '</ul></nav>';
        container.innerHTML = html;
    }
    
    goToPage(page) {
        this.currentPage = page;
        this.loadObservations();
    }
    
    selectObservation(id) {
        const obs = this.data.find(row => row[this.idColumn] == id);
        if (!obs) return;
        
        this.selectedObservation = obs;
        
        // Update UI
        this.updateObservationInfo(obs);
        this.updateNoteContent(obs);
        this.updateSummaryContent(obs);
        this.updateAssignedConcepts(obs);
        
        // Update selected state in list
        document.querySelectorAll('.observation-row').forEach(row => {
            row.classList.toggle('selected', row.dataset.id == id);
        });
        
        // Enable export button
        document.getElementById('exportSelectedEncounterBtn').disabled = false;
        
        // Show sections
        document.getElementById('observationInfo').style.display = 'block';
        document.getElementById('noteViewer').style.display = 'block';
        if (this.config.features.show_summaries && obs[this.summaryColumn]) {
            document.getElementById('summarySection').style.display = 'block';
        }
    }
    
    updateObservationInfo(obs) {
        const container = document.getElementById('observationMetadata');
        if (!container) return;
        
        let html = '';
        this.metadataColumns.forEach(col => {
            if (obs[col] !== undefined) {
                const displayName = this.config.ui.column_display_names?.[col] || col;
                let value = obs[col];
                
                // Round LOS values to 2 decimal places
                if (col === 'length_of_stay_days' || col === 'global_median_los') {
                    value = parseFloat(value).toFixed(2);
                }
                
                html += `
                    <div class="metadata-item">
                        <span class="text-muted">${displayName}:</span>
                        <span class="metadata-value ms-2">${value}</span>
                    </div>
                `;
            }
        });
        
        container.innerHTML = html;
    }
    
    updateNoteContent(obs) {
        const container = document.getElementById('noteContent');
        if (!container) return;
        
        const noteText = obs[this.noteColumn] || 'No note content available';
        container.textContent = noteText;
        
        // Clear any existing selections
        this.clearTextSelections();
    }
    
    updateSummaryContent(obs) {
        if (!this.config.features.show_summaries) return;
        
        const container = document.getElementById('summaryContent');
        if (!container) return;
        
        const summaryText = obs[this.summaryColumn] || '';
        if (summaryText) {
            // Check if the summary contains markdown-like formatting
            if (this.containsMarkdown(summaryText)) {
                // Use marked library to render markdown if available, otherwise simple HTML conversion
                if (typeof marked !== 'undefined') {
                    container.innerHTML = marked.parse(summaryText);
                } else {
                    // Simple markdown-to-HTML conversion for basic formatting
                    container.innerHTML = this.simpleMarkdownToHtml(summaryText);
                }
            } else {
                // Plain text
                container.textContent = summaryText;
            }
            document.getElementById('summarySection').style.display = 'block';
        } else {
            document.getElementById('summarySection').style.display = 'none';
        }
    }
    
    containsMarkdown(text) {
        // Check for common markdown patterns
        const markdownPatterns = [
            /^#{1,6}\s+/m,      // Headers
            /\*\*.*?\*\*/,      // Bold
            /\*.*?\*/,          // Italic
            /^[-*+]\s+/m,       // Lists
            /^\d+\.\s+/m,       // Numbered lists
            /\[.*?\]\(.*?\)/,   // Links
            /`.*?`/,            // Code
            /^>\s+/m            // Blockquotes
        ];
        
        return markdownPatterns.some(pattern => pattern.test(text));
    }
    
    simpleMarkdownToHtml(text) {
        return text
            // Headers
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            // Bold and italic
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Lists (simple conversion)
            .replace(/^[-*+]\s+(.*$)/gm, '<li>$1</li>')
            .replace(/^\d+\.\s+(.*$)/gm, '<li>$1</li>')
            // Wrap consecutive list items in ul/ol tags
            .replace(/(<li>.*?<\/li>)/gs, (match) => {
                return '<ul>' + match + '</ul>';
            })
            // Line breaks
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            // Wrap in paragraphs
            .replace(/^(.*)$/gm, '<p>$1</p>')
            // Clean up empty paragraphs
            .replace(/<p><\/p>/g, '')
            // Clean up nested paragraph tags
            .replace(/<p>(<[^>]+>)/g, '$1')
            .replace(/(<\/[^>]+>)<\/p>/g, '$1');
    }
    
    updateAssignedConcepts(obs) {
        if (!this.config.features.show_concepts) return;
        
        this.initializations.forEach(init => {
            const container = document.getElementById(`conceptsList-${init.id}`);
            if (!container) return;
            
            let html = '';
            
            init.concepts.forEach((concept, index) => {
                const colName = `${init.concept_prefix}_concept_${index}`;
                const value = obs[colName] || 0;
                const isAssigned = value > 0.5;
                
                html += `
                    <div class="concept-card ${isAssigned ? 'active' : ''}">
                        <div class="d-flex justify-content-between align-items-center">
                            <span>${concept}</span>
                            <span class="badge ${isAssigned ? 'bg-success' : 'bg-secondary'}">
                                ${value.toFixed(2)}
                            </span>
                        </div>
                    </div>
                `;
            });
            
            if (html === '') {
                html = '<div class="text-muted">No concepts for this initialization</div>';
            }
            
            container.innerHTML = html;
        });
    }
    
    setupTextSelection() {
        // Add selection handling for note and summary
        ['noteContent', 'summaryContent'].forEach(elementId => {
            const element = document.getElementById(elementId);
            if (!element) return;
            
            element.addEventListener('mouseup', () => {
                const selection = window.getSelection();
                if (selection.toString().trim()) {
                    this.handleTextSelection(selection, elementId);
                }
            });
        });
    }
    
    handleTextSelection(selection, sourceElement) {
        const selectedText = selection.toString().trim();
        if (!selectedText) return;
        
        // Create annotation
        const annotation = {
            id: this.annotationIdCounter++,
            text: selectedText,
            source: sourceElement.includes('note') ? 'note' : 'summary',
            observationId: this.selectedObservation[this.idColumn],
            timestamp: new Date().toISOString()
        };
        
        this.annotations.push(annotation);
        this.updateAnnotationsList();
        
        // Clear selection
        selection.removeAllRanges();
    }
    
    clearTextSelections() {
        // Clear visual highlights if implemented
    }
    
    updateAnnotationsList() {
        const container = document.getElementById('selectedExcerptsList');
        if (!container) return;
        
        if (this.annotations.length === 0) {
            container.innerHTML = '<div class="text-muted text-center py-3">No excerpts selected yet</div>';
            return;
        }
        
        let html = '';
        this.annotations.forEach(annotation => {
            const sourceClass = annotation.source === 'note' ? 'from-note' : 'from-summary';
            html += `
                <div class="excerpt-item ${sourceClass}">
                    <div class="excerpt-source">
                        From ${annotation.source} - ${this.config.ui.encounter_display_name} ${annotation.observationId}
                    </div>
                    <div class="excerpt-text">${annotation.text}</div>
                    <button class="btn btn-sm btn-outline-danger mt-1" 
                            onclick="app.removeAnnotation(${annotation.id})">
                        Remove
                    </button>
                </div>
            `;
        });
        
        container.innerHTML = html;
        
        // Enable export button
        document.getElementById('exportAnnotationsBtn').disabled = false;
    }
    
    removeAnnotation(id) {
        this.annotations = this.annotations.filter(a => a.id !== id);
        this.updateAnnotationsList();
    }
    
    clearAllAnnotations() {
        if (confirm('Are you sure you want to clear all annotations?')) {
            this.annotations = [];
            this.updateAnnotationsList();
        }
    }
    
    exportObservations() {
        // Export filtered data as CSV
        const headers = Object.keys(this.filteredData[0] || {});
        const csv = [
            headers.join(','),
            ...this.filteredData.map(row => 
                headers.map(h => JSON.stringify(row[h] || '')).join(',')
            )
        ].join('\n');
        
        this.downloadFile(csv, 'filtered_observations.csv', 'text/csv');
    }
    
    exportSelectedEncounter() {
        if (!this.selectedObservation) return;
        
        // Create a detailed export of the selected encounter
        const exportData = {
            observation: this.selectedObservation,
            annotations: this.annotations.filter(a => a.observationId === this.selectedObservation[this.idColumn]),
            exportDate: new Date().toISOString(),
            config: {
                title: this.config.ui.title,
                initializations: this.initializations.map(init => init.name)
            }
        };
        
        const json = JSON.stringify(exportData, null, 2);
        const filename = `encounter_${this.selectedObservation[this.idColumn]}_export.json`;
        this.downloadFile(json, filename, 'application/json');
    }
    
    exportAnnotations(includeNotes) {
        if (this.annotations.length === 0) {
            alert('No annotations to export');
            return;
        }
        
        // Group annotations by observation
        const groupedAnnotations = {};
        this.annotations.forEach(annotation => {
            if (!groupedAnnotations[annotation.observationId]) {
                groupedAnnotations[annotation.observationId] = [];
            }
            groupedAnnotations[annotation.observationId].push(annotation);
        });
        
        // Create export data
        const exportData = {
            exportDate: new Date().toISOString(),
            totalAnnotations: this.annotations.length,
            observations: Object.keys(groupedAnnotations).map(obsId => {
                const obs = this.data.find(row => row[this.idColumn] == obsId);
                const result = {
                    observationId: obsId,
                    annotations: groupedAnnotations[obsId]
                };
                
                if (includeNotes && obs) {
                    result.noteContent = obs[this.noteColumn];
                    result.summaryContent = obs[this.summaryColumn];
                    result.metadata = {};
                    this.metadataColumns.forEach(col => {
                        if (obs[col] !== undefined) {
                            result.metadata[col] = obs[col];
                        }
                    });
                }
                
                return result;
            })
        };
        
        const json = JSON.stringify(exportData, null, 2);
        const filename = `annotations_export_${new Date().toISOString().split('T')[0]}.json`;
        this.downloadFile(json, filename, 'application/json');
    }
    
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Utility function for debouncing
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize the application
let app;
document.addEventListener('DOMContentLoaded', () => {
    window.viewer = new StandaloneConceptAnalysisInterface();
    app = window.viewer; // Keep backward compatibility
});