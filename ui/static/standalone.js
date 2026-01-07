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
        this.renderAllConcepts();

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

        let html = `
            <div class="enhanced-flow-diagram">
                <h3>Concept Bottleneck Model Training Process</h3>

                <!-- Binary Concepts Emphasis -->
                <div class="binary-emphasis">
                    <h4><i class="fas fa-question-circle me-2"></i>All Concepts Are Binary Questions</h4>
                    <p>Every concept extracted by our method is framed as a yes/no question that can be answered from clinical notes</p>
                </div>

                <div class="flow-steps">
                    <div class="flow-step-enhanced has-feedback" data-step="1">
                        <div class="feedback-badge">1</div>
                        <div class="step-number">1</div>
                        <div class="step-content">
                            <h6>Raw Clinical Notes</h6>
                            <p>Unstructured admission notes containing all clinical information about the patient's hospitalization</p>
                        </div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-step-enhanced has-feedback" data-step="2">
                        <div class="feedback-badge">2</div>
                        <div class="step-number">2</div>
                        <div class="step-content">
                            <h6>Hospital Course Summary</h6>
                            <p>LLM generates structured summaries focusing on events and factors that may influence length of stay</p>
                        </div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-step-enhanced has-feedback" data-step="3">
                        <div class="step-number">3</div>
                        <div class="step-content">
                            <h6>Initial Concept Extraction</h6>
                            <p>LLM extracts initial clinical concepts as binary questions from the generated summaries</p>
                        </div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-step-enhanced" data-step="4">
                        <div class="step-number">4</div>
                        <div class="step-content">
                            <h6>Baseline Initialization</h6>
                            <p>Statistical analysis identifies most predictive features to seed the initial concept set</p>
                        </div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-step-enhanced has-feedback" data-step="5">
                        <div class="feedback-badge">3</div>
                        <div class="step-number">5</div>
                        <div class="step-content">
                            <h6>Iterative Refinement</h6>
                            <p>Greedy optimization generates new concepts and refines existing ones to improve model performance</p>
                        </div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-step-enhanced" data-step="6">
                        <div class="step-number">6</div>
                        <div class="step-content">
                            <h6>Final Binary Concepts</h6>
                            <p>Set of interpretable yes/no questions that predict length of stay outcomes</p>
                        </div>
                    </div>
                </div>

                <!-- Key Insight Box -->
                <div class="mt-4 p-3" style="background-color: #e7f3ff; border-left: 4px solid #0d6efd; border-radius: 0.5rem;">
                    <h5 style="color: #0d6efd; margin-bottom: 1rem;"><i class="fas fa-lightbulb me-2"></i>Key Insight</h5>
                    <p class="mb-0" style="font-size: 1.1rem;">By generating binary concepts, our method produces highly interpretable models where each prediction can be traced back to specific yes/no questions about the patient's clinical course.</p>
                </div>
            </div>
        `;

        // Add summary statistics
        if (this.initializations.length > 0) {
            html += `
                <div class="training-summary mt-4">
                    <h6>Training Results</h6>
                    <div class="row">
            `;

            this.initializations.forEach((init, index) => {
                const auc = init.final_auc || 'N/A';
                const iterations = init.num_iterations || 'N/A';
                html += `
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title text-center">${init.name}</h6>
                                <div class="text-center mb-3">
                                    <strong>Final AUC:</strong> ${typeof auc === 'number' ? auc.toFixed(3) : auc}<br>
                                    <strong>Binary Concepts:</strong> ${init.concepts ? init.concepts.length : 0}<br>
                                    <strong>Training Iterations:</strong> ${iterations}
                                </div>
                                <div id="dendrogram-${index}" class="dendrogram-container">
                                    <h6 class="text-center mb-2">Concept Relationships</h6>
                                    <div class="text-center text-muted">Loading analysis...</div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;

        // Generate dendrograms for each initialization
        this.generateDendrograms();

        // Load feedback opportunities and concept examples
        this.loadOverviewSections();
    }

    renderFeedbackOpportunities() {
        const container = document.getElementById('feedbackOpportunitiesContent');
        if (!container) return;

        const html = `
            <div class="feedback-opportunities">
                <h3 style="text-align: center; margin-bottom: 2rem; color: #2c3e50;">Where Your Clinical Expertise Can Improve Our Method</h3>

                <p style="font-size: 1.2rem; text-align: center; margin-bottom: 3rem; color: #495057;">
                    We've identified three key areas where clinician feedback can significantly enhance concept quality and model performance:
                </p>

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

                <!-- Feedback Point 2: Concept Phrasing -->
                <div class="feedback-section">
                    <div class="feedback-icon">2</div>
                    <h5><i class="fas fa-edit me-2"></i>Initial Concept Extraction & Phrasing</h5>

                    <div class="feedback-what">
                        <strong><i class="fas fa-question-circle me-2"></i>What feedback do we need?</strong>
                        <p>Guide us on how to phrase concepts to be clinically meaningful, actionable, and interpretable for healthcare providers.</p>
                    </div>

                    <div class="feedback-how">
                        <strong><i class="fas fa-tools me-2"></i>How does this help?</strong>
                        <p>Ensures concepts are framed in ways that make clinical sense and provide actionable insights rather than just statistical correlations.</p>
                    </div>

                    <div class="feedback-example">
                        <strong><i class="fas fa-lightbulb me-2"></i>Example transformation needed:</strong>
                        <div class="before-after mt-3">
                            <div class="before">
                                <strong>Current phrasing:</strong><br>
                                "Does the note mention the patient needing home health services?"
                            </div>
                            <div class="arrow-transform">→</div>
                            <div class="after">
                                <strong>Better phrasing:</strong><br>
                                "Does the note mention that discharge is delayed due to pending home health services approval?"
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Feedback Point 3: Iterative Refinement -->
                <div class="feedback-section">
                    <div class="feedback-icon">3</div>
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

                <!-- Call to Action -->
                <div class="mt-4 p-4" style="background-color: #f0f9ff; border: 2px solid #0284c7; border-radius: 1rem; text-align: center;">
                    <h4 style="color: #0284c7; margin-bottom: 1rem;"><i class="fas fa-handshake me-2"></i>Your Expertise Makes the Difference</h4>
                    <p style="font-size: 1.1rem; color: #0369a1; margin: 0;">
                        By providing feedback in these areas, you help us build more clinically relevant, interpretable, and actionable predictive models for length of stay.
                    </p>
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    renderConceptExamples() {
        const container = document.getElementById('conceptExamplesContent');
        if (!container) return;

        const html = `
            <div class="concept-examples">
                <h3 style="text-align: center; margin-bottom: 2rem; color: #2c3e50;">Real Concept Examples from Sepsis Length of Stay Study</h3>

                <p style="font-size: 1.2rem; text-align: center; margin-bottom: 3rem; color: #495057;">
                    Here are actual binary concepts extracted by our method, showing how clinical factors become yes/no questions:
                </p>

                <!-- Example 1: Before/After Transformation -->
                <div class="example-transformation">
                    <h5><i class="fas fa-exchange-alt me-2"></i>Concept Evolution: Home Health Services</h5>
                    <p>This example shows how clinical feedback can transform a basic concept into a more actionable one:</p>

                    <div class="before-after">
                        <div class="before">
                            <strong>Initial Extraction:</strong><br>
                            "Does the note mention the patient needing home health services?"<br><br>
                            <small><strong>Issue:</strong> Too general - doesn't capture the impact on length of stay</small>
                        </div>
                        <div class="arrow-transform">
                            <i class="fas fa-arrow-right"></i><br>
                            <small style="color: #0d6efd;">Clinical Feedback</small>
                        </div>
                        <div class="after">
                            <strong>Refined Concept:</strong><br>
                            "Does the note mention that discharge is delayed due to pending home health services approval?"<br><br>
                            <small><strong>Better:</strong> Captures the specific relationship to length of stay</small>
                        </div>
                    </div>
                </div>

                <!-- Current Concepts from Sepsis Study -->
                <div class="mt-4">
                    <h5><i class="fas fa-list-check me-2"></i>Current Binary Concepts (Sample)</h5>
                    <p>These are actual concepts extracted from our sepsis length of stay experiment:</p>

                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header bg-success text-white">
                                    <h6 class="mb-0"><i class="fas fa-check-circle me-2"></i>Well-Formed Concepts</h6>
                                </div>
                                <div class="card-body">
                                    <ul class="list-unstyled">
                                        <li class="mb-2"><i class="fas fa-question-circle text-success me-2"></i>Does the note mention the patient undergoing surgical intervention?</li>
                                        <li class="mb-2"><i class="fas fa-question-circle text-success me-2"></i>Does the note mention the patient experiencing a delay in discharge?</li>
                                        <li class="mb-2"><i class="fas fa-question-circle text-success me-2"></i>Does the note mention the patient requiring multiple consultations?</li>
                                        <li class="mb-2"><i class="fas fa-question-circle text-success me-2"></i>Does the note mention the patient having COVID-19 pneumonia?</li>
                                    </ul>
                                    <small class="text-muted">These concepts are specific and clinically interpretable</small>
                                </div>
                            </div>
                        </div>

                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header bg-warning text-dark">
                                    <h6 class="mb-0"><i class="fas fa-exclamation-triangle me-2"></i>Could Be Improved</h6>
                                </div>
                                <div class="card-body">
                                    <ul class="list-unstyled">
                                        <li class="mb-2">
                                            <i class="fas fa-question-circle text-warning me-2"></i>
                                            Does the note mention the patient needing home health services?
                                            <small class="d-block text-muted">→ Could specify "delayed discharge due to pending HHS"</small>
                                        </li>
                                        <li class="mb-2">
                                            <i class="fas fa-question-circle text-warning me-2"></i>
                                            Does the note mention the patient experiencing discharge planning barriers?
                                            <small class="d-block text-muted">→ Could be more specific about the type of barrier</small>
                                        </li>
                                    </ul>
                                    <small class="text-muted">These could benefit from clinical feedback for clarity</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Interactive Feedback Exercise -->
                <div class="mt-4 p-4" style="background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 1rem;">
                    <h5 style="color: #856404;"><i class="fas fa-comment-dots me-2"></i>Try This: Clinical Feedback Exercise</h5>
                    <p style="color: #856404;">For each concept below, consider:</p>
                    <ul style="color: #856404;">
                        <li>Is this concept clear and specific enough to be clinically useful?</li>
                        <li>Does it capture the factor's relationship to length of stay?</li>
                        <li>How would you rephrase it to be more actionable?</li>
                    </ul>

                    <div class="mt-3 p-3" style="background-color: white; border-radius: 0.5rem;">
                        <div class="mb-3">
                            <strong>Example Concept:</strong> "Does the note mention the patient experiencing homelessness impact?"<br>
                            <div class="mt-2">
                                <button class="btn btn-sm btn-outline-primary me-2" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">Show Our Thoughts</button>
                                <div style="display: none;" class="alert alert-info mt-2">
                                    This could be refined to: "Does the note mention that discharge is complicated by homelessness or lack of stable housing?" - This better captures the impact on discharge planning.
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Concept Impact Visualization -->
                <div class="mt-4">
                    <h5><i class="fas fa-chart-line me-2"></i>How Concepts Drive Predictions</h5>
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h4 class="text-success">+0.847</h4>
                                    <p class="mb-0">Surgical intervention</p>
                                    <small class="text-muted">Strong positive predictor of longer stay</small>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h4 class="text-success">+0.623</h4>
                                    <p class="mb-0">Multiple consultations</p>
                                    <small class="text-muted">Moderate positive predictor</small>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h4 class="text-danger">-0.432</h4>
                                    <p class="mb-0">Discharge against medical advice</p>
                                    <small class="text-muted">Negative predictor (shorter stay)</small>
                                </div>
                            </div>
                        </div>
                    </div>
                    <p class="text-center mt-3 text-muted">
                        <i class="fas fa-info-circle me-2"></i>
                        Each binary concept contributes to the final length of stay prediction with an interpretable coefficient
                    </p>
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    loadOverviewSections() {
        // Load feedback opportunities
        this.renderFeedbackOpportunities();

        // Load concept examples
        this.renderConceptExamples();
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

    // Display static dendrogram images generated during export
    generateDendrograms() {
        this.initializations.forEach((init, index) => {
            const trainingHistory = this.methodInfo.training_histories?.[init.concept_prefix];
            const container = document.getElementById(`dendrogram-${index}`);
            
            if (!container) return;
            
            // Check if we have a static dendrogram image
            if (trainingHistory && trainingHistory.dendrogram_image) {
                container.innerHTML = `
                    <h6 class="text-center mb-2">Concept Clustering</h6>
                    <div class="text-center">
                        <img src="${trainingHistory.dendrogram_image}" 
                             alt="Dendrogram for ${init.name}" 
                             style="max-width: 100%; height: auto; border: 1px solid #dee2e6; border-radius: 0.375rem;"
                             onerror="this.parentElement.innerHTML='<div class=\\'text-muted\\'>Dendrogram image not found</div>'">
                    </div>
                `;
            } else {
                // Fallback to clustering data rendering if no image is available
                if (trainingHistory && trainingHistory.dendrogram_data) {
                    this.renderDendrogramFromClusteringData(
                        trainingHistory.dendrogram_data, 
                        `dendrogram-${index}`, 
                        init.name
                    );
                } else {
                    container.innerHTML = '<div class="text-center text-muted">No clustering data available</div>';
                }
            }
        });
    }
    
    // Calculate Pearson correlation coefficient between two arrays
    pearsonCorrelation(x, y) {
        const n = x.length;
        if (n !== y.length || n === 0) return 0;
        
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        if (denominator === 0) return 0;
        return numerator / denominator;
    }
    
    // Distance function (1 - |correlation|)
    correlationDistance(x, y) {
        try {
            // Check for constant vectors
            const varX = this.variance(x);
            const varY = this.variance(y);
            
            if (varX === 0 || varY === 0) {
                return x.every((val, i) => val === y[i]) ? 0 : 1;
            }
            
            const corr = this.pearsonCorrelation(x, y);
            return 1 - Math.abs(corr);
        } catch (e) {
            return 1; // Maximum distance on error
        }
    }
    
    variance(array) {
        const mean = array.reduce((a, b) => a + b, 0) / array.length;
        return array.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / array.length;
    }
    
    // Single linkage clustering
    singleLinkage(distances) {
        const n = distances.length;
        const clusters = Array.from({ length: n }, (_, i) => ({ indices: [i], height: 0 }));
        const linkage = [];
        
        for (let step = 0; step < n - 1; step++) {
            let minDist = Infinity;
            let minI = -1, minJ = -1;
            
            // Find minimum distance between clusters
            for (let i = 0; i < clusters.length; i++) {
                for (let j = i + 1; j < clusters.length; j++) {
                    if (clusters[i] && clusters[j]) {
                        let clusterDist = Infinity;
                        
                        // Single linkage: minimum distance between any two points
                        for (const idx1 of clusters[i].indices) {
                            for (const idx2 of clusters[j].indices) {
                                clusterDist = Math.min(clusterDist, distances[idx1][idx2]);
                            }
                        }
                        
                        if (clusterDist < minDist) {
                            minDist = clusterDist;
                            minI = i;
                            minJ = j;
                        }
                    }
                }
            }
            
            if (minI === -1 || minJ === -1) break;
            
            // Merge clusters
            const newCluster = {
                indices: [...clusters[minI].indices, ...clusters[minJ].indices],
                height: minDist,
                left: clusters[minI],
                right: clusters[minJ]
            };
            
            linkage.push({
                left: clusters[minI],
                right: clusters[minJ],
                distance: minDist,
                size: newCluster.indices.length
            });
            
            // Replace clusters with merged cluster
            clusters[minI] = newCluster;
            clusters[minJ] = null;
            clusters.splice(minJ, 1);
        }
        
        return linkage;
    }
    
    renderDendrogramFromClusteringData(clusteringData, containerId, modelName) {
        const container = document.getElementById(containerId);
        if (!container) {
            return;
        }
        
        console.log(`Creating dendrogram for ${modelName} with clustering data:`, clusteringData);
        
        if (!clusteringData || !clusteringData.concepts || clusteringData.concepts.length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No concepts available</div>';
            return;
        }
        
        if (!clusteringData.has_clustering || clusteringData.concepts.length === 1) {
            // Special handling for single concept or no clustering
            const concept = clusteringData.concepts[0];
            container.innerHTML = `
                <div class="p-2">
                    <div class="d-flex align-items-start">
                        <div style="font-size: 11px; line-height: 1.3;">
                            <div>${concept.name}</div>
                            <div style="font-size: 10px; color: #666; margin-top: 2px;">
                                (${(concept.posterior_probability).toFixed(2)})
                            </div>
                        </div>
                    </div>
                </div>
            `;
            return;
        }
        
        try {
            // Render horizontal dendrogram from linkage matrix
            this.renderHorizontalDendrogram(clusteringData, containerId);
            
        } catch (error) {
            console.error('Error creating dendrogram:', error);
            container.innerHTML = `<div class="text-center text-muted">Clustering visualization unavailable</div>`;
        }
    }
    
    renderHorizontalDendrogram(clusteringData, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = ''; // Clear existing content
        
        const concepts = clusteringData.concepts;
        const linkage = clusteringData.linkage;
        
        // Set up dimensions - use full container width with better measurements
        const containerRect = container.getBoundingClientRect();
        const containerWidth = containerRect.width || container.clientWidth || container.offsetWidth || 600;
        const margin = { top: 20, right: Math.min(150, containerWidth * 0.25), bottom: 20, left: 20 };
        const height = Math.max(300, concepts.length * 30);
        const width = Math.max(500, containerWidth - margin.left - margin.right);
        
        const svg = d3.select(`#${containerId}`)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Convert scipy linkage matrix to D3 hierarchy
        const tree = this.linkageToTree(linkage, concepts);
        if (!tree) {
            container.innerHTML = '<div class="text-center text-muted">Unable to build dendrogram</div>';
            return;
        }
        
        // Create D3 hierarchy and tree layout
        const root = d3.hierarchy(tree);
        const treeLayout = d3.tree().size([height, width - margin.right]);
        treeLayout(root);
        
        // Create scales for distances - use more of the available width
        const maxDistance = d3.max(linkage, d => d[2]) || 1;
        const xScale = d3.scaleLinear()
            .domain([0, maxDistance])
            .range([0, width * 0.85]); // Use 85% of width for better spacing and text room
        
        // Render links
        g.selectAll('.link')
            .data(root.links())
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('d', d => {
                // Create horizontal dendrogram links
                const source = d.source;
                const target = d.target;
                return `M${xScale(source.data.distance || 0)},${source.x}
                        L${xScale(target.data.distance || 0)},${source.x}
                        L${xScale(target.data.distance || 0)},${target.x}`;
            })
            .style('fill', 'none')
            .style('stroke', '#ccc')
            .style('stroke-width', '1px');
        
        // Render nodes
        const nodes = g.selectAll('.node')
            .data(root.descendants())
            .enter()
            .append('g')
            .attr('class', 'node')
            .attr('transform', d => `translate(${xScale(d.data.distance || 0)},${d.x})`);
        
        // Add circles for internal nodes
        nodes.filter(d => d.children)
            .append('circle')
            .attr('r', 3)
            .style('fill', '#999');
        
        // Add labels for leaf nodes
        nodes.filter(d => !d.children && d.data.concept)
            .append('text')
            .attr('x', 8)
            .attr('dy', '.35em')
            .style('font-size', '11px')
            .style('fill', '#333')
            .text(d => `${d.data.concept.name} (${d.data.concept.posterior_probability.toFixed(2)})`);
        
        // Add x-axis for distance scale
        const xAxis = d3.axisBottom(xScale)
            .ticks(5)
            .tickSize(-height);
        
        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis)
            .selectAll('text')
            .style('font-size', '10px')
            .style('fill', '#666');
        
        // Style the axis
        g.selectAll('.axis line')
            .style('stroke', '#ddd')
            .style('stroke-width', '0.5px');
        
        g.selectAll('.axis path')
            .style('stroke', '#ddd');
    }
    
    linkageToTree(linkage, concepts) {
        if (!linkage || linkage.length === 0) {
            return null;
        }
        
        const n = concepts.length;
        const nodes = {};
        
        // Create leaf nodes
        concepts.forEach((concept, i) => {
            nodes[i] = {
                concept: concept,
                distance: 0,
                isLeaf: true
            };
        });
        
        // Process linkage matrix to build tree
        linkage.forEach((link, i) => {
            const [left, right, distance, count] = link;
            const nodeId = n + i;
            
            nodes[nodeId] = {
                distance: distance,
                children: [nodes[left], nodes[right]],
                isLeaf: false
            };
        });
        
        // Return the root (last node created)
        const rootId = n + linkage.length - 1;
        return nodes[rootId];
    }
    
    renderSimpleDendrogram(concepts, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = ''; // Clear existing content
        
        // Increased dimensions for better readability
        const margin = { top: 20, right: 20, bottom: 20, left: 20 };
        const width = 260;
        const nodeHeight = 40; // More space per node
        const height = Math.max(200, concepts.length * nodeHeight);
        
        // Create scrollable container if needed
        const scrollContainer = d3.select(`#${containerId}`)
            .style('max-height', '300px')
            .style('overflow-y', 'auto')
            .style('overflow-x', 'hidden');
        
        const svg = scrollContainer
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Sort concepts by coefficient magnitude for visualization
        const sortedConcepts = [...concepts].sort((a, b) => 
            Math.abs(b.coefficient || 0) - Math.abs(a.coefficient || 0)
        );
        
        // For small sets, just show a simple list
        if (sortedConcepts.length <= 2) {
            // Simple vertical list for 1-2 concepts
            sortedConcepts.forEach((concept, i) => {
                const y = i * nodeHeight + 20;
                const conceptG = g.append('g')
                    .attr('transform', `translate(10, ${y})`);
                
                // Add colored circle based on coefficient
                const coef = concept.coefficient || 0;
                conceptG.append('circle')
                    .attr('r', 4)
                    .style('fill', coef > 0 ? '#28a745' : '#dc3545');
                
                // Add concept text with wrapping
                const text = concept.text || 'Unknown concept';
                const wrappedText = self.wrapText(text, 220);
                
                const textG = conceptG.append('text')
                    .attr('x', 10)
                    .attr('y', 0)
                    .style('font-size', '11px')
                    .style('fill', '#333');
                
                wrappedText.forEach((line, lineIndex) => {
                    textG.append('tspan')
                        .attr('x', 10)
                        .attr('dy', lineIndex === 0 ? 0 : '1.1em')
                        .text(line);
                });
                
                // Add coefficient and posterior probability
                const statsG = conceptG.append('g')
                    .attr('transform', `translate(10, ${(wrappedText.length * 12) + 5})`);
                
                statsG.append('text')
                    .style('font-size', '10px')
                    .style('fill', coef > 0 ? '#28a745' : '#dc3545')
                    .style('font-weight', 'bold')
                    .text(`Coef: ${coef.toFixed(3)}`);
                
                if (concept.posterior_probability !== undefined) {
                    statsG.append('text')
                        .attr('y', 12)
                        .style('font-size', '10px')
                        .style('fill', '#666')
                        .text(`Prob: ${(concept.posterior_probability * 100).toFixed(0)}%`);
                }
            });
            return;
        }
        
        // Create hierarchical structure for larger sets
        const hierarchyData = this.buildHierarchy([], sortedConcepts);
        
        if (!hierarchyData) {
            container.innerHTML = '<div class="text-center text-muted">Unable to build concept hierarchy</div>';
            return;
        }
        
        const root = d3.hierarchy(hierarchyData);
        const tree = d3.tree().size([height - 40, width - 100]);
        
        tree(root);
        
        // Add links
        g.selectAll('.dendrogram-link')
            .data(root.links())
            .enter()
            .append('path')
            .attr('class', 'dendrogram-link')
            .attr('d', d3.linkHorizontal()
                .x(d => d.y)
                .y(d => d.x))
            .style('fill', 'none')
            .style('stroke', '#ccc')
            .style('stroke-width', '1px');
        
        // Add nodes
        const node = g.selectAll('.dendrogram-node')
            .data(root.descendants())
            .enter()
            .append('g')
            .attr('class', 'dendrogram-node')
            .attr('transform', d => `translate(${d.y},${d.x})`);
        
        // Add circles for all nodes
        node.append('circle')
            .attr('r', 3)
            .style('fill', d => {
                if (d.children) return '#999';
                const coef = d.data.concept?.coefficient || 0;
                return coef > 0 ? '#28a745' : '#dc3545';
            });
        
        // Add text labels for leaf nodes with full text
        const leafNodes = node.filter(d => !d.children && d.data.concept);
        
        // Store reference to this for use in callback
        const self = this;
        
        leafNodes.each(function(d) {
            const concept = d.data.concept;
            const text = concept.text || 'Unknown concept';
            const coef = concept.coefficient || 0;
            
            // Create a group for the text
            const textGroup = d3.select(this)
                .append('g')
                .attr('transform', 'translate(8, 0)');
            
            // Add background for better readability
            const textBg = textGroup.append('rect')
                .attr('x', -2)
                .attr('y', -10)
                .attr('rx', 2)
                .style('fill', 'white')
                .style('opacity', 0.8);
            
            // Add wrapped text
            const textEl = textGroup.append('text')
                .style('font-size', '10px')
                .style('fill', '#333');
            
            const wrappedText = self.wrapText(text, 180);
            wrappedText.forEach((line, i) => {
                textEl.append('tspan')
                    .attr('x', 0)
                    .attr('dy', i === 0 ? 0 : '1.1em')
                    .text(line);
            });
            
            // Add coefficient
            textEl.append('tspan')
                .attr('x', 0)
                .attr('dy', '1.3em')
                .style('font-weight', 'bold')
                .style('fill', coef > 0 ? '#28a745' : '#dc3545')
                .text(`Coef: ${coef > 0 ? '+' : ''}${coef.toFixed(3)}`);
            
            // Add posterior probability if available
            const postProb = concept.posterior_probability;
            if (postProb !== undefined) {
                textEl.append('tspan')
                    .attr('x', 0)
                    .attr('dy', '1.1em')
                    .style('font-size', '9px')
                    .style('fill', '#666')
                    .text(`Prob: ${(postProb * 100).toFixed(0)}%`);
            }
            
            // Update background size
            const bbox = textEl.node().getBBox();
            textBg
                .attr('width', bbox.width + 4)
                .attr('height', bbox.height + 4);
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
    
    renderAllConcepts() {
        const container = document.getElementById('allConceptsContent');
        if (!container || this.initializations.length === 0) return;
        
        // Check if any initialization has model-specific concept analysis
        const hasModelAnalysis = this.initializations.some(init => {
            const trainingHistory = this.methodInfo.training_histories?.[init.concept_prefix];
            return trainingHistory && trainingHistory.model_concept_analysis;
        });
        
        if (!hasModelAnalysis) {
            container.innerHTML = `
                <div class="text-center p-4">
                    <h6>Model-Specific Concept Analysis</h6>
                    <p class="text-muted">Coefficient analysis not available. Ensure training data and extraction files are present during export.</p>
                </div>
            `;
            return;
        }
        
        // Show explanation first
        let html = `
            <div class="mb-4">
                <h5>Model-Specific Concept Analysis</h5>
                <div class="alert alert-info">
                    <p class="mb-2"><strong>Per-Model Analysis:</strong> Each model's concepts are analyzed independently to avoid collinearity issues.</p>
                    <ul class="mb-0">
                        <li><strong>Positive coefficients (green):</strong> When this concept is present, the patient is more likely to have a longer length of stay</li>
                        <li><strong>Negative coefficients (red):</strong> When this concept is present, the patient is more likely to have a shorter length of stay</li>
                        <li><strong>Magnitude:</strong> Larger absolute values indicate stronger predictive power within that model</li>
                    </ul>
                </div>
            </div>
            
            <div class="row">
        `;
        
        // Create a card for each model
        this.initializations.forEach(init => {
            const trainingHistory = this.methodInfo.training_histories?.[init.concept_prefix];
            if (trainingHistory && trainingHistory.model_concept_analysis) {
                const analysis = trainingHistory.model_concept_analysis;
                const cbmAuc = trainingHistory.final_auc || 'N/A';
                const modelAuc = analysis.model_specific_auc || 'N/A';
                const concepts = analysis.concepts || [];
                
                html += `
                    <div class="col-md-4 mb-4">
                        <div class="card h-100">
                            <div class="card-header">
                                <h6 class="mb-0">${init.name} - Coefficient Analysis</h6>
                                <small class="text-muted">
                                    CBM AUC: ${typeof cbmAuc === 'number' ? cbmAuc.toFixed(3) : cbmAuc} | 
                                    Model AUC: ${typeof modelAuc === 'number' ? modelAuc.toFixed(3) : modelAuc}
                                </small>
                            </div>
                            <div class="card-body p-2">
                                <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
                                    <table class="table table-sm table-striped">
                                        <thead class="table-dark sticky-top">
                                            <tr>
                                                <th style="width: 70%">Concept</th>
                                                <th style="width: 30%" class="text-center">Coefficient</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                `;
                
                // Sort concepts by absolute coefficient (descending)
                const sortedConcepts = [...concepts].sort((a, b) => b.abs_coefficient - a.abs_coefficient);
                
                sortedConcepts.forEach(concept => {
                    const cleanText = concept.text.replace(/^Does the note mention (that )?the patient /, '').replace(/\?$/, '');
                    const coef = concept.coefficient;
                    
                    // Color coding
                    let colorClass = '';
                    let iconHtml = '';
                    if (coef > 0) {
                        colorClass = 'text-success';
                        iconHtml = '<span class="text-success me-1" style="font-size: 0.9em;">↑</span>';
                    } else {
                        colorClass = 'text-danger';
                        iconHtml = '<span class="text-danger me-1" style="font-size: 0.9em;">↓</span>';
                    }
                    
                    html += `
                        <tr>
                            <td style="font-size: 0.8rem; line-height: 1.2;">
                                ${iconHtml}${cleanText}
                            </td>
                            <td class="${colorClass} fw-bold text-center" style="font-size: 0.8rem;">
                                ${coef > 0 ? '+' : ''}${coef.toFixed(3)}
                            </td>
                        </tr>
                    `;
                });
                
                html += `
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
        });
        
        html += `
            </div>
        `;
        
        container.innerHTML = html;
    }
    
    
    renderPromptsSection() {
        if (!this.methodInfo.prompts) return '';

        let html = `
            <div class="mt-4">
                <h3 style="text-align: center; margin-bottom: 2rem; color: #2c3e50;">Prompts & Feedback Integration Points</h3>
                <p class="text-muted mb-4" style="font-size: 1.2rem; line-height: 1.6; text-align: center;">
                    These prompts guide our LLM at each step. <strong>Your clinical feedback directly shapes these prompts</strong> to improve concept quality and clinical relevance.
                </p>

                <div class="alert alert-info mb-4">
                    <h5><i class="fas fa-info-circle me-2"></i>How Feedback Gets Integrated</h5>
                    <p>Your clinical insights help us modify these prompts in specific ways:</p>
                    <ul class="mb-0">
                        <li><strong>Examples:</strong> We add examples of good vs. poor concept phrasing</li>
                        <li><strong>Focus Areas:</strong> We emphasize clinical factors you identify as important</li>
                        <li><strong>Terminology:</strong> We adjust language to match clinical conventions</li>
                        <li><strong>Context:</strong> We provide better context about length of stay relationships</li>
                    </ul>
                </div>
        `;

        const promptCategories = {
            'step1_summary': {
                title: 'Step 1: Hospital Course Summary Generation',
                feedback: 'Your feedback helps us identify which clinical events and processes most impact length of stay',
                icon: 'fas fa-clipboard-list'
            },
            'step2_extraction': {
                title: 'Step 2: Initial Concept Extraction',
                feedback: 'Your expertise guides how we phrase concepts to be clinically meaningful and actionable',
                icon: 'fas fa-search-plus'
            },
            'step3_initialization': {
                title: 'Step 3: Baseline Concept Initialization',
                feedback: 'Statistical analysis - limited direct feedback integration at this step',
                icon: 'fas fa-chart-bar'
            },
            'step4_generation': {
                title: 'Step 4: Iterative Concept Generation & Refinement',
                feedback: 'Your insights help identify missing concepts and improve existing ones',
                icon: 'fas fa-sync-alt'
            }
        };

        // Display prompts in the correct order matching CBM training steps
        const orderedSteps = ['step1_summary', 'step2_extraction', 'step3_initialization', 'step4_generation'];

        orderedSteps.forEach(category => {
            const prompts = this.methodInfo.prompts[category];
            if (!prompts || prompts.length === 0) return;

            const categoryInfo = promptCategories[category] || { title: category, feedback: '', icon: 'fas fa-cog' };
            const hasFeedback = category !== 'step3_initialization';

            html += `
                <div class="prompt-category mb-4">
                    <div class="d-flex align-items-center mb-3">
                        <h6 class="text-primary border-bottom pb-2 flex-grow-1">
                            <i class="${categoryInfo.icon} me-2"></i>${categoryInfo.title}
                        </h6>
                        ${hasFeedback ? '<span class="badge bg-warning text-dark ms-2"><i class="fas fa-comments me-1"></i>Feedback Integration</span>' : ''}
                    </div>

                    ${hasFeedback ? `
                    <div class="alert alert-warning">
                        <strong><i class="fas fa-user-md me-2"></i>Clinical Feedback Integration:</strong>
                        ${categoryInfo.feedback}
                    </div>
                    ` : ''}
            `;

            prompts.forEach((prompt, index) => {
                const collapseId = `collapse-${category}-${index}`;
                html += `
                    <div class="card mb-2">
                        <div class="card-header ${hasFeedback ? 'bg-warning bg-opacity-10' : 'bg-light'}">
                            <button class="btn btn-link btn-sm text-decoration-none fw-bold p-0" type="button"
                                    data-bs-toggle="collapse" data-bs-target="#${collapseId}"
                                    aria-expanded="false" aria-controls="${collapseId}">
                                <i class="fas fa-chevron-right me-2" id="icon-${collapseId}"></i>
                                ${prompt.filename}
                                ${hasFeedback ? '<i class="fas fa-edit ms-2 text-warning" title="Modifiable based on clinical feedback"></i>' : ''}
                            </button>
                        </div>
                        <div id="${collapseId}" class="collapse">
                            <div class="card-body p-3">
                                ${hasFeedback ? `
                                <div class="alert alert-info mb-3">
                                    <small><strong><i class="fas fa-lightbulb me-1"></i>Feedback Integration Points:</strong>
                                    This prompt can be modified based on your clinical expertise to improve concept quality and relevance.</small>
                                </div>
                                ` : ''}
                                <div class="prompt-text-container">
                                    <pre class="prompt-text-full">${this.escapeHtml(prompt.content)}</pre>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += `</div>`;
        });

        html += `
                <div class="mt-4 p-4" style="background-color: #e7f3ff; border: 2px solid #0d6efd; border-radius: 1rem; text-align: center;">
                    <h5 style="color: #0d6efd; margin-bottom: 1rem;"><i class="fas fa-handshake me-2"></i>Collaborative Improvement</h5>
                    <p style="font-size: 1.1rem; color: #0369a1; margin: 0;">
                        Each prompt represents an opportunity for clinical expertise to enhance our method's performance and interpretability.
                    </p>
                </div>
            </div>
        `;
        return html;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
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
        this.clearTextSelections('note');
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
            const assignedCol = `assigned_concepts_${init.concept_prefix}`;
            const assignedConcepts = obs[assignedCol] ? obs[assignedCol].split(',').map(c => c.trim()) : [];
            
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
    
    clearTextSelections(source) {
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
    app = new StandaloneConceptAnalysisInterface();
});