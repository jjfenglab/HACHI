/**
 * Generic Concept Analysis Interface
 * Configurable JavaScript for dynamic UI functionality
 */
class GenericConceptAnalysisInterface {
    constructor() {
        this.currentPage = 1;
        this.currentSearch = '';
        this.partitionFilter = 'all';
        this.conceptFilter = '';
        this.config = null;
        this.selectedObservation = null;
        this.selectedNote = null;
        this.generatedSummary = null;
        this.availableConcepts = [];
        this.assignedConcepts = [];
        this.annotations = [];
        this.annotationIdCounter = 0;
        this.currentSelection = null;
        this.selectionToolbarVisible = false;
        
        this.init();
    }
    
    async init() {
        try {
            // Load configuration first
            await this.loadConfig();
            
            // Initialize UI based on configuration
            this.initializeUI();
            
            // Bind events
            this.bindEvents();
            
            // Load initial data
            await this.loadHealthStatus();
            await this.loadConcepts();
            await this.loadObservations();
            
            // Load prompts if LLM is available
            if (this.config.features.llm) {
                await this.loadPrompts();
            }
            
            // Setup text selection
            this.setupTextSelection();
            
            // Initialize button states
            this.updateExportEncounterButtonState();
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showError('observationsList', 'Failed to initialize application: ' + error.message);
        }
    }
    
    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            if (!response.ok) {
                throw new Error('Failed to load configuration');
            }
            this.config = await response.json();
            console.log('Configuration loaded:', this.config);
        } catch (error) {
            console.error('Error loading configuration:', error);
            throw error;
        }
    }
    
    initializeUI() {
        // Update header title
        document.title = this.config.ui.title;
        
        // Show/hide partition filter based on feature availability
        const partitionFilterContainer = document.getElementById('partitionFilterContainer');
        if (this.config.features.train_test_split) {
            partitionFilterContainer.style.display = 'block';
        }
        
        // Show/hide concept filter based on feature availability
        const conceptFilterContainer = document.getElementById('conceptFilterContainer');
        if (this.config.features.concepts) {
            conceptFilterContainer.style.display = 'block';
        }
        
        // Show/hide summary generator based on LLM availability
        const summaryGenerator = document.getElementById('summaryGenerator');
        if (this.config.features.llm) {
            summaryGenerator.style.display = 'block';
        }
        
        // Update placeholder text based on configuration
        const searchInput = document.getElementById('searchInput');
        const encounterName = this.config.ui.encounter_display_name || 'observations';
        searchInput.placeholder = `Search ${encounterName.toLowerCase()}...`;
        
        // Update note column title with cleaned name
        const noteColumnTitle = document.getElementById('noteColumnTitle');
        if (noteColumnTitle && this.config.dataset.note_column) {
            const columnDisplayNames = this.config.ui.column_display_names || {};
            const displayName = columnDisplayNames[this.config.dataset.note_column] || 
                              this.cleanDisplayName(this.config.dataset.note_column);
            noteColumnTitle.textContent = displayName;
        }
        
        // Initialize resizable note container
        this.initializeResizableNote();
    }
    
    bindEvents() {
        // Search
        const searchInput = document.getElementById('searchInput');
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.currentSearch = e.target.value;
                this.currentPage = 1;
                this.loadObservations();
            }, 500);
        });
        
        // Partition filter
        if (this.config.features.train_test_split) {
            document.getElementById('partitionFilter').addEventListener('change', (e) => {
                this.partitionFilter = e.target.value;
                this.currentPage = 1;
                this.loadObservations();
            });
        }
        
        // Concept filter
        if (this.config.features.concepts) {
            document.getElementById('conceptFilter').addEventListener('change', (e) => {
                this.conceptFilter = e.target.value;
                this.currentPage = 1;
                this.loadObservations();
            });
        }
        
        // Summary generation (if LLM available)
        if (this.config.features.llm) {
            document.getElementById('promptSelect').addEventListener('change', (e) => {
                this.updateGenerateButtonState();
            });
            
            document.getElementById('generateSummaryBtn').addEventListener('click', () => {
                this.generateSummary();
            });
        }
        
        // Clear selections
        document.getElementById('clearNoteSelections').addEventListener('click', () => {
            this.clearAnnotations('note');
        });
        
        document.getElementById('clearSummarySelections').addEventListener('click', () => {
            this.clearAnnotations('summary');
        });
        
        document.getElementById('clearAllAnnotationsBtn').addEventListener('click', () => {
            this.clearAllAnnotations();
        });
        
        // Selection toolbar
        document.getElementById('linkSelectionBtn').addEventListener('click', () => {
            this.linkSelection();
        });
        
        document.getElementById('cancelSelectionBtn').addEventListener('click', () => {
            this.hideSelectionToolbar();
        });
        
        // Export
        document.getElementById('exportObservationsBtn').addEventListener('click', () => {
            this.exportObservations();
        });
        
        document.getElementById('exportSelectedEncounterBtn').addEventListener('click', () => {
            this.exportSelectedEncounter();
        });
    }
    
    initializeResizableNote() {
        const noteContent = document.getElementById('noteContent');
        const resizeHandle = document.querySelector('.resize-handle');
        
        if (!noteContent || !resizeHandle) return;
        
        let isResizing = false;
        let startY = 0;
        let startHeight = 0;
        
        resizeHandle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startY = e.clientY;
            startHeight = noteContent.offsetHeight;
            
            // Prevent text selection while resizing
            document.body.style.userSelect = 'none';
            document.body.style.cursor = 'ns-resize';
            
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            const deltaY = e.clientY - startY;
            const newHeight = startHeight + deltaY;
            
            // Respect min and max heights from CSS
            if (newHeight >= 200 && newHeight <= 800) {
                noteContent.style.height = newHeight + 'px';
            }
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.userSelect = '';
                document.body.style.cursor = '';
            }
        });
        
        // Also handle touch events for mobile
        resizeHandle.addEventListener('touchstart', (e) => {
            isResizing = true;
            startY = e.touches[0].clientY;
            startHeight = noteContent.offsetHeight;
            e.preventDefault();
        });
        
        document.addEventListener('touchmove', (e) => {
            if (!isResizing) return;
            
            const deltaY = e.touches[0].clientY - startY;
            const newHeight = startHeight + deltaY;
            
            if (newHeight >= 200 && newHeight <= 800) {
                noteContent.style.height = newHeight + 'px';
            }
        });
        
        document.addEventListener('touchend', () => {
            isResizing = false;
        });
    }
    
    async loadHealthStatus() {
        try {
            const response = await fetch('/api/health');
            const status = await response.json();
            
            const statusDiv = document.getElementById('healthStatus');
            statusDiv.innerHTML = `
                <div>DB: ${status.db_rows} rows</div>
                <div>Concepts: ${status.concepts_available}</div>
                <div>LLM: ${status.llm_api_available ? 'Ready' : 'Unavailable'}</div>
            `;
        } catch (error) {
            console.error('Error loading health status:', error);
        }
    }
    
    async loadConcepts() {
        if (!this.config.features.concepts) return;
        
        try {
            const response = await fetch('/api/concepts');
            const data = await response.json();
            if (response.ok) {
                this.availableConcepts = data.concepts || [];
                this.renderConceptSelect();
                this.updateAnnotationConceptSelection();
            }
        } catch (error) {
            console.error('Failed to load concepts:', error);
        }
    }
    
    renderConceptSelect() {
        const conceptFilter = document.getElementById('conceptFilter');
        if (!conceptFilter) return;
        
        conceptFilter.innerHTML = '<option value="">All Concepts</option>';
        
        this.availableConcepts.forEach(concept => {
            const cleanConcept = this.cleanConceptText(concept);
            const option = document.createElement('option');
            option.value = concept;
            option.textContent = cleanConcept;
            option.title = concept;
            conceptFilter.appendChild(option);
        });
    }
    
    cleanConceptText(concept) {
        // Remove common prefixes to make concepts more readable
        let cleaned = concept;
        
        const prefixesToRemove = [
            'Does the note mention the patient requiring ',
            'Does the note mention the patient having ',
            'Does the note mention the patient ',
            'Does the note mention ',
            'Does the patient have ',
            'Does the patient require ',
            'Is the patient ',
            'Was the patient '
        ];
        
        for (const prefix of prefixesToRemove) {
            if (cleaned.toLowerCase().startsWith(prefix.toLowerCase())) {
                cleaned = cleaned.substring(prefix.length);
                break;
            }
        }
        
        // Remove trailing question mark and capitalize first letter
        cleaned = cleaned.replace(/\?$/, '');
        cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
        
        return cleaned;
    }
    
    cleanDisplayName(name) {
        // Convert underscore/snake_case names to readable format
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    async loadObservations() {
        const listDiv = document.getElementById('observationsList');
        listDiv.innerHTML = '<div class="loading">Loading observations...</div>';
        
        try {
            const params = new URLSearchParams({
                page: this.currentPage,
                per_page: 20,
                search: this.currentSearch,
                partition: this.partitionFilter,
                concept: this.conceptFilter
            });
            
            const response = await fetch(`/api/observations?${params}`);
            const data = await response.json();
            
            if (response.ok) {
                this.renderObservations(data.observations);
                this.renderPagination(data);
            } else {
                this.showError('observationsList', data.error || 'Failed to load observations');
            }
        } catch (error) {
            this.showError('observationsList', 'Network error loading observations');
            console.error('Error loading observations:', error);
        }
    }
    
    renderObservations(observations) {
        const listDiv = document.getElementById('observationsList');
        
        if (observations.length === 0) {
            listDiv.innerHTML = '<div class="text-muted text-center py-3">No observations found</div>';
            return;
        }
        
        const idColumn = this.config.dataset.id_column;
        const displayColumns = this.config.ui.display_columns || [];
        const columnDisplayNames = this.config.ui.column_display_names || {};
        
        listDiv.innerHTML = observations.map(observation => {
            const isSelected = this.selectedObservation && 
                             this.selectedObservation[idColumn] === observation[idColumn];
            
            // Build metadata display
            let metadataHtml = '';
            displayColumns.forEach(column => {
                if (observation[column] !== undefined) {
                    const displayName = columnDisplayNames[column] || column;
                    let value = observation[column];
                    
                    // Format length_of_stay to 2 decimal places
                    if (column.toLowerCase().includes('length_of_stay') && !isNaN(value)) {
                        value = parseFloat(value).toFixed(2);
                    }
                    
                    metadataHtml += `<div class="small text-muted">${displayName}: ${value}</div>`;
                }
            });
            
            // Add partition badge if available
            let partitionBadge = '';
            if (observation.partition) {
                const badgeClass = observation.partition === 'train' ? 'bg-primary' : 'bg-info';
                partitionBadge = `<span class="badge ${badgeClass} info-badge">${observation.partition.toUpperCase()}</span>`;
            }
            
            return `
                <div class="observation-row p-2 border-bottom ${isSelected ? 'selected' : ''}" 
                     onclick="app.selectObservation(${observation[idColumn]}, this)">
                    <div class="fw-bold">${this.config.ui.encounter_display_name}: ${observation[idColumn]}</div>
                    ${metadataHtml}
                    ${partitionBadge ? `<div class="mt-1">${partitionBadge}</div>` : ''}
                </div>
            `;
        }).join('');
    }
    
    renderPagination(data) {
        const paginationDiv = document.getElementById('pagination');
        
        if (data.total_count <= data.per_page) {
            paginationDiv.innerHTML = '';
            return;
        }
        
        const totalPages = Math.ceil(data.total_count / data.per_page);
        let html = '<nav><ul class="pagination pagination-sm justify-content-center">';
        
        // Previous button
        html += `<li class="page-item ${!data.has_prev ? 'disabled' : ''}">
                    <a class="page-link" href="#" onclick="app.goToPage(${data.page - 1})">&laquo;</a>
                 </li>`;
        
        // Page numbers
        const startPage = Math.max(1, data.page - 2);
        const endPage = Math.min(totalPages, data.page + 2);
        
        for (let i = startPage; i <= endPage; i++) {
            html += `<li class="page-item ${i === data.page ? 'active' : ''}">
                        <a class="page-link" href="#" onclick="app.goToPage(${i})">${i}</a>
                     </li>`;
        }
        
        // Next button
        html += `<li class="page-item ${!data.has_next ? 'disabled' : ''}">
                    <a class="page-link" href="#" onclick="app.goToPage(${data.page + 1})">&raquo;</a>
                 </li>`;
        
        html += '</ul></nav>';
        paginationDiv.innerHTML = html;
    }
    
    goToPage(page) {
        if (page < 1) return;
        this.currentPage = page;
        this.loadObservations();
    }
    
    async selectObservation(observationId, clickedElement = null) {
        try {
            const response = await fetch(`/api/observation/${observationId}`);
            const data = await response.json();
            
            if (response.ok) {
                this.selectedObservation = data;
                this.selectedNote = data[this.config.dataset.note_column];
                this.renderObservationInfo();
                this.renderNote();
                this.clearAllAnnotations();
                
                // Update UI state
                this.updateGenerateButtonState();
                this.updateExportEncounterButtonState();
                
                // Update visual selection
                document.querySelectorAll('.observation-row').forEach(row => {
                    row.classList.remove('selected');
                });
                
                if (clickedElement) {
                    const observationRow = clickedElement.closest('.observation-row');
                    if (observationRow) {
                        observationRow.classList.add('selected');
                    }
                }
                
                // Load concepts for this observation if available
                if (this.config.features.concepts) {
                    await this.loadObservationConcepts(observationId);
                }
            } else {
                this.showError('noteContent', data.error || 'Failed to load observation');
            }
        } catch (error) {
            this.showError('noteContent', 'Network error loading observation');
            console.error('Error loading observation:', error);
        }
    }
    
    renderObservationInfo() {
        if (!this.selectedObservation) return;
        
        const metadataDiv = document.getElementById('observationMetadata');
        const idColumn = this.config.dataset.id_column;
        const displayColumns = this.config.ui.display_columns || [];
        const columnDisplayNames = this.config.ui.column_display_names || {};
        
        let html = `<div class="metadata-item">
                        <strong>${columnDisplayNames[idColumn] || this.cleanDisplayName(idColumn)}:</strong> 
                        <span class="metadata-value">${this.selectedObservation[idColumn]}</span>
                    </div>`;
        
        displayColumns.forEach(column => {
            if (this.selectedObservation[column] !== undefined) {
                const displayName = columnDisplayNames[column] || this.cleanDisplayName(column);
                let value = this.selectedObservation[column];
                
                // Format length_of_stay to 2 decimal places
                if (column.toLowerCase().includes('length_of_stay') && !isNaN(value)) {
                    value = parseFloat(value).toFixed(2);
                }
                
                html += `<div class="metadata-item">
                            <strong>${displayName}:</strong> 
                            <span class="metadata-value">${value}</span>
                         </div>`;
            }
        });
        
        // Add partition info if available
        if (this.selectedObservation.partition) {
            html += `<div class="metadata-item">
                        <strong>Partition:</strong> 
                        <span class="badge ${this.selectedObservation.partition === 'train' ? 'bg-primary' : 'bg-info'} metadata-value">
                            ${this.selectedObservation.partition.toUpperCase()}
                        </span>
                     </div>`;
        }
        
        metadataDiv.innerHTML = html;
        
        // Find the card body container
        const cardBody = document.querySelector('#observationInfo .card-body');
        
        // Remove existing concepts container if any
        const existingConceptsContainer = cardBody.querySelector('.assigned-concepts-container');
        if (existingConceptsContainer) {
            existingConceptsContainer.remove();
        }
        
        // Add assigned concepts if available
        if (this.selectedObservation.assigned_concepts && this.selectedObservation.assigned_concepts.length > 0) {
            const conceptsContainer = document.createElement('div');
            conceptsContainer.className = 'assigned-concepts-container';
            conceptsContainer.innerHTML = `
                <h6 class="mb-2">CBM-Assigned Concepts</h6>
                <div class="assigned-concepts-list">
                    ${this.selectedObservation.assigned_concepts.map(concept => {
                        const cleanConcept = this.cleanConceptText(concept);
                        return `<span class="badge bg-warning text-dark assigned-concept-badge" title="${concept}">${cleanConcept}</span>`;
                    }).join('')}
                </div>
            `;
            cardBody.appendChild(conceptsContainer);
        }
        
        document.getElementById('observationInfo').style.display = 'block';
    }
    
    renderNote() {
        const noteContent = document.getElementById('noteContent');
        noteContent.textContent = this.selectedNote || 'Select an observation to view the content';
        document.getElementById('noteViewer').style.display = this.selectedNote ? 'block' : 'none';
        
        // Clear any existing annotations
        this.renderAnnotations();
    }
    
    async loadObservationConcepts(observationId) {
        try {
            // The concepts are already loaded with the observation data
            if (this.selectedObservation && this.selectedObservation.assigned_concepts) {
                this.assignedConcepts = this.selectedObservation.assigned_concepts;
                this.renderAssignedConcepts();
            } else {
                this.assignedConcepts = [];
                this.renderAssignedConcepts();
            }
        } catch (error) {
            console.error('Error loading observation concepts:', error);
        }
    }
    
    renderAssignedConcepts() {
        const assignedConceptsList = document.getElementById('assignedConceptsList');
        const assignedConceptsSection = document.getElementById('assignedConceptsSection');
        
        if (this.assignedConcepts.length === 0) {
            assignedConceptsSection.style.display = 'none';
            return;
        }
        
        assignedConceptsSection.style.display = 'block';
        assignedConceptsList.innerHTML = this.assignedConcepts.map(concept => `
            <div class="concept-card" data-concept="${concept.concept}">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <span class="concept-badge badge concept-${concept.color_index}" title="${concept.concept}">
                            ${this.cleanConceptText(concept.concept)}
                        </span>
                    </div>
                    <small class="text-muted">Positive</small>
                </div>
            </div>
        `).join('');
    }
    
    updateAnnotationConceptSelection() {
        const conceptSelection = document.getElementById('conceptSelection');
        if (!conceptSelection) return;
        
        conceptSelection.innerHTML = '<option value="">Select concept...</option>';
        
        this.availableConcepts.forEach(concept => {
            const option = document.createElement('option');
            option.value = concept;
            option.textContent = this.cleanConceptText(concept);
            option.title = concept;
            conceptSelection.appendChild(option);
        });
    }
    
    async loadPrompts() {
        if (!this.config.features.llm) return;
        
        try {
            const response = await fetch('/api/prompts');
            const data = await response.json();
            
            if (response.ok) {
                this.renderPromptSelect(data.prompts);
            }
        } catch (error) {
            console.error('Failed to load prompts:', error);
        }
    }
    
    renderPromptSelect(prompts) {
        const promptSelect = document.getElementById('promptSelect');
        if (!promptSelect) return;
        
        promptSelect.innerHTML = '<option value="">Select a prompt...</option>';
        
        prompts.forEach(prompt => {
            const option = document.createElement('option');
            option.value = prompt;
            option.textContent = prompt;
            promptSelect.appendChild(option);
        });
    }
    
    updateGenerateButtonState() {
        const generateBtn = document.getElementById('generateSummaryBtn');
        if (!generateBtn) return;
        
        const promptSelect = document.getElementById('promptSelect');
        const hasPrompt = promptSelect && promptSelect.value;
        const hasNote = this.selectedNote;
        
        generateBtn.disabled = !hasPrompt || !hasNote;
    }
    
    async generateSummary() {
        if (!this.config.features.llm || !this.selectedNote) return;
        
        const promptSelect = document.getElementById('promptSelect');
        const generateBtn = document.getElementById('generateSummaryBtn');
        const summaryStatus = document.getElementById('summaryStatus');
        const summarySection = document.getElementById('summarySection');
        const summaryContent = document.getElementById('summaryContent');
        
        if (!promptSelect.value) return;
        
        generateBtn.disabled = true;
        summaryStatus.textContent = 'Generating summary...';
        
        try {
            const response = await fetch('/api/llm/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: await this.getPromptContent(promptSelect.value),
                    note: this.selectedNote
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.generatedSummary = data.response;
                
                // Render summary as markdown
                if (typeof marked !== 'undefined') {
                    summaryContent.innerHTML = marked.parse(this.generatedSummary);
                } else {
                    summaryContent.textContent = this.generatedSummary;
                }
                
                summarySection.style.display = 'block';
                summaryStatus.textContent = 'Summary generated successfully';
                
                // Re-setup text selection for the newly generated summary
                setTimeout(() => {
                    const updatedSummaryContent = document.getElementById('summaryContent');
                    this.setupTextSelectionForElement(updatedSummaryContent);
                }, 100);
                
                // Clear any existing summary annotations
                this.clearAnnotations('summary');
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            summaryStatus.textContent = `Error: ${error.message}`;
        } finally {
            generateBtn.disabled = false;
        }
    }
    
    async getPromptContent(filename) {
        try {
            const response = await fetch(`/api/prompt/${filename}`);
            const data = await response.json();
            return response.ok ? data.content : '';
        } catch (error) {
            console.error('Error loading prompt content:', error);
            return '';
        }
    }
    
    setupTextSelection() {
        const noteContent = document.getElementById('noteContent');
        const summaryContent = document.getElementById('summaryContent');
        
        this.setupTextSelectionForElement(noteContent);
        this.setupTextSelectionForElement(summaryContent);
        
        // Hide toolbar when clicking elsewhere
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#selectionToolbar') && !e.target.closest('.annotatable-text')) {
                this.hideSelectionToolbar();
            }
        });
    }
    
    setupTextSelectionForElement(element) {
        if (!element) return;
        
        const mouseUpHandler = (e) => {
            this.handleTextSelection(e);
        };
        
        element.removeEventListener('mouseup', mouseUpHandler);
        element.addEventListener('mouseup', mouseUpHandler);
    }
    
    handleTextSelection(event) {
        const selection = window.getSelection();
        const selectedText = selection.toString().trim();
        
        if (selectedText.length === 0) {
            this.hideSelectionToolbar();
            return;
        }
        
        const range = selection.getRangeAt(0);
        const container = event.currentTarget;
        const source = container.dataset.source;
        
        // Calculate position for toolbar
        const rect = range.getBoundingClientRect();
        
        this.currentSelection = {
            text: selectedText,
            source: source,
            startOffset: range.startOffset,
            endOffset: range.endOffset,
            containerText: container.textContent
        };
        
        this.showSelectionToolbar(rect.left + window.scrollX, rect.bottom + window.scrollY);
    }
    
    showSelectionToolbar(x, y) {
        const toolbar = document.getElementById('selectionToolbar');
        if (!toolbar) return;
        
        // Position toolbar within viewport
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
        
        let viewportX = x - scrollLeft;
        let viewportY = y - scrollTop;
        
        let adjustedX = Math.max(10, Math.min(viewportX, viewportWidth - 350));
        let adjustedY = Math.max(10, Math.min(viewportY, viewportHeight - 60));
        
        if (viewportY > viewportHeight - 120) {
            adjustedY = Math.max(10, viewportY - 70);
        }
        
        toolbar.style.left = adjustedX + 'px';
        toolbar.style.top = adjustedY + 'px';
        toolbar.style.display = 'block';
        this.selectionToolbarVisible = true;
        
        const conceptSelection = document.getElementById('conceptSelection');
        if (conceptSelection) {
            conceptSelection.value = '';
        }
    }
    
    hideSelectionToolbar() {
        document.getElementById('selectionToolbar').style.display = 'none';
        this.selectionToolbarVisible = false;
        this.currentSelection = null;
        window.getSelection().removeAllRanges();
    }
    
    linkSelection() {
        if (!this.currentSelection) return;
        
        const conceptSelection = document.getElementById('conceptSelection');
        const selectedConcept = conceptSelection.value;
        
        if (!selectedConcept) {
            alert('Please select a concept to link to this text.');
            return;
        }
        
        const annotation = {
            id: ++this.annotationIdCounter,
            source: this.currentSelection.source,
            concept: selectedConcept,
            text: this.currentSelection.text,
            startOffset: this.currentSelection.startOffset,
            endOffset: this.currentSelection.endOffset,
            colorIndex: this.getConceptColorIndex(selectedConcept)
        };
        
        this.annotations.push(annotation);
        this.renderAnnotations();
        this.renderSelectedExcerpts();
        this.updateExportButton();
        this.hideSelectionToolbar();
    }
    
    getConceptColorIndex(concept) {
        const conceptIndex = this.assignedConcepts.findIndex(c => c.concept === concept);
        return conceptIndex >= 0 ? this.assignedConcepts[conceptIndex].color_index : this.annotations.length % 8;
    }
    
    renderAnnotations() {
        this.renderSelectedExcerpts();
    }
    
    renderSelectedExcerpts() {
        const selectedExcerptsList = document.getElementById('selectedExcerptsList');
        
        if (this.annotations.length === 0) {
            selectedExcerptsList.innerHTML = `
                <div class="text-muted text-center py-3">
                    No excerpts selected yet
                </div>
            `;
            return;
        }
        
        // Group annotations by concept
        const grouped = {};
        this.annotations.forEach(annotation => {
            if (!grouped[annotation.concept]) {
                grouped[annotation.concept] = [];
            }
            grouped[annotation.concept].push(annotation);
        });
        
        selectedExcerptsList.innerHTML = Object.entries(grouped).map(([concept, annotations]) => `
            <div class="mb-3">
                <h6 class="concept-badge badge concept-${annotations[0].colorIndex} mb-2" title="${concept}">
                    ${this.cleanConceptText(concept)}
                </h6>
                ${annotations.map(annotation => `
                    <div class="excerpt-item ${annotation.source === 'note' ? 'from-note' : 'from-summary'}">
                        <div class="excerpt-source">
                            <i class="fas ${annotation.source === 'note' ? 'fa-file-medical' : 'fa-file-alt'}"></i>
                            From ${annotation.source === 'note' ? 'Note' : 'Summary'}
                            <button class="btn btn-sm btn-outline-danger float-end" onclick="app.removeAnnotation(${annotation.id})">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                        <div class="excerpt-text">"${annotation.text}"</div>
                    </div>
                `).join('')}
            </div>
        `).join('');
    }
    
    removeAnnotation(annotationId) {
        this.annotations = this.annotations.filter(a => a.id !== annotationId);
        this.renderAnnotations();
        this.updateExportButton();
    }
    
    clearAnnotations(source) {
        if (source) {
            this.annotations = this.annotations.filter(a => a.source !== source);
        } else {
            this.annotations = [];
        }
        this.renderAnnotations();
        this.updateExportButton();
    }
    
    clearAllAnnotations() {
        this.annotations = [];
        this.renderSelectedExcerpts();
        this.updateExportButton();
    }
    
    updateExportButton() {
        const exportBtn = document.getElementById('exportAnnotationsBtn');
        if (exportBtn) {
            exportBtn.disabled = this.annotations.length === 0;
        }
    }
    
    async exportObservations() {
        try {
            const response = await fetch('/api/export/observations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    search: this.currentSearch,
                    partition: this.partitionFilter,
                    concept: this.conceptFilter
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'filtered_observations.csv';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                const data = await response.json();
                alert('Export failed: ' + data.error);
            }
        } catch (error) {
            alert('Export failed: ' + error.message);
        }
    }
    
    updateExportEncounterButtonState() {
        const exportBtn = document.getElementById('exportSelectedEncounterBtn');
        if (exportBtn) {
            exportBtn.disabled = !this.selectedObservation;
        }
    }
    
    exportSelectedEncounter() {
        if (!this.selectedObservation) {
            alert('Please select an encounter first');
            return;
        }
        
        // Generate export data for the selected encounter
        const exportData = {
            observation: this.selectedObservation,
            note: this.selectedNote,
            generatedSummary: this.generatedSummary,
            exportDate: new Date().toISOString(),
            config: this.config
        };
        
        // Generate HTML export
        const html = this.generateEncounterHTML(exportData);
        
        // Create and download file
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const idColumn = this.config.dataset.id_column;
        const encounterId = this.selectedObservation[idColumn];
        const filename = `encounter_${encounterId}_${new Date().toISOString().split('T')[0]}.html`;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    generateEncounterHTML(data) {
        const idColumn = data.config.dataset.id_column;
        const noteColumn = data.config.dataset.note_column;
        const displayColumns = data.config.ui.display_columns || [];
        const columnDisplayNames = data.config.ui.column_display_names || {};
        
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Encounter Export - ${data.observation[idColumn]}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metadata-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .metadata-item { padding: 0.5rem; border: 1px solid #dee2e6; border-radius: 0.375rem; }
        .note-content { background-color: #f8f9fa; padding: 1rem; border-radius: 0.375rem; white-space: pre-wrap; font-family: 'Courier New', monospace; line-height: 1.5; }
        .summary-content { background-color: #f0f9ff; padding: 1rem; border-radius: 0.375rem; white-space: pre-wrap; line-height: 1.6; margin-top: 2rem; }
        .concepts-list { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
        .concept-badge { background-color: #ffc107; color: #000; padding: 0.35rem 0.65rem; border-radius: 0.375rem; font-size: 0.8rem; }
        .export-header { border-bottom: 2px solid #dee2e6; padding-bottom: 1rem; margin-bottom: 2rem; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="export-header">
            <h1>Encounter Export</h1>
            <p class="text-muted">Exported on: ${new Date(data.exportDate).toLocaleString()}</p>
            <p class="text-muted">Encounter ID: ${data.observation[idColumn]}</p>
        </div>
        
        <div class="metadata-section mb-4">
            <h2>Encounter Details</h2>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>${columnDisplayNames[idColumn] || idColumn}:</strong><br>
                    ${data.observation[idColumn]}
                </div>
                ${displayColumns.map(column => {
                    if (data.observation[column] !== undefined) {
                        const displayName = columnDisplayNames[column] || column;
                        let value = data.observation[column];
                        // Format length_of_stay if needed
                        if (column.toLowerCase().includes('length_of_stay') && !isNaN(value)) {
                            value = parseFloat(value).toFixed(2);
                        }
                        return `
                        <div class="metadata-item">
                            <strong>${displayName}:</strong><br>
                            ${value}
                        </div>`;
                    }
                    return '';
                }).join('')}
                ${data.observation.partition ? `
                <div class="metadata-item">
                    <strong>Partition:</strong><br>
                    <span class="badge ${data.observation.partition === 'train' ? 'bg-primary' : 'bg-info'}">${data.observation.partition.toUpperCase()}</span>
                </div>` : ''}
            </div>
            
            ${data.observation.assigned_concepts && data.observation.assigned_concepts.length > 0 ? `
            <div class="concepts-section mt-3">
                <h5>CBM-Assigned Concepts</h5>
                <div class="concepts-list">
                    ${data.observation.assigned_concepts.map(concept => `
                        <span class="concept-badge">${concept.replace(/^Does the note mention (the patient |the patient )?/, '').replace(/\?$/, '')}</span>
                    `).join('')}
                </div>
            </div>` : ''}
        </div>
        
        <div class="note-section mb-4">
            <h2>${columnDisplayNames[noteColumn] || noteColumn}</h2>
            <div class="note-content">${data.note || 'No note content available'}</div>
        </div>
        
        ${data.generatedSummary ? `
        <div class="summary-section">
            <h2>Generated Summary</h2>
            <div class="summary-content">${data.generatedSummary}</div>
        </div>` : ''}
    </div>
</body>
</html>
        `;
    }
    
    exportAnnotations(includeNoteContent = true) {
        if (this.annotations.length === 0) return;
        
        // Generate export data
        const exportData = {
            observation: this.selectedObservation,
            annotations: this.annotations,
            generatedSummary: this.generatedSummary,
            selectedNote: includeNoteContent ? this.selectedNote : null,
            exportDate: new Date().toISOString(),
            includeNoteContent: includeNoteContent,
            config: this.config
        };
        
        // Generate HTML export
        const html = this.generateAnnotationHTML(exportData);
        
        // Create and download file
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const filename = includeNoteContent ? 
            `concept_validation_full_${this.selectedObservation[this.config.dataset.id_column]}_${new Date().toISOString().split('T')[0]}.html` :
            `concept_validation_annotations_${this.selectedObservation[this.config.dataset.id_column]}_${new Date().toISOString().split('T')[0]}.html`;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    generateAnnotationHTML(data) {
        const idColumn = data.config.dataset.id_column;
        const noteColumn = data.config.dataset.note_column;
        
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Concept Validation - ${data.observation[idColumn]}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .concept-0 { background-color: rgba(255, 235, 59, 0.4); border-color: #fbc02d; }
        .concept-1 { background-color: rgba(76, 175, 80, 0.4); border-color: #4caf50; }
        .concept-2 { background-color: rgba(33, 150, 243, 0.4); border-color: #2196f3; }
        .concept-3 { background-color: rgba(255, 87, 34, 0.4); border-color: #ff5722; }
        .concept-4 { background-color: rgba(156, 39, 176, 0.4); border-color: #9c27b0; }
        .concept-5 { background-color: rgba(0, 188, 212, 0.4); border-color: #00bcd4; }
        .concept-6 { background-color: rgba(255, 152, 0, 0.4); border-color: #ff9800; }
        .concept-7 { background-color: rgba(96, 125, 139, 0.4); border-color: #607d8b; }
        .note-text { font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.4; white-space: pre-wrap; }
        .summary-text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 14px; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <header class="mb-4">
            <h1>${data.config.ui.title} - Validation Report</h1>
            <p class="text-muted">Generated on ${new Date(data.exportDate).toLocaleString()}</p>
        </header>
        
        <div class="row mb-4">
            <div class="col-md-6">
                <h3>Observation Information</h3>
                <table class="table table-sm">
                    <tr><th>${data.config.ui.column_display_names[idColumn] || idColumn}:</th><td>${data.observation[idColumn]}</td></tr>
                    ${(data.config.ui.display_columns || []).map(column => 
                        `<tr><th>${data.config.ui.column_display_names[column] || column}:</th><td>${data.observation[column] || 'N/A'}</td></tr>`
                    ).join('')}
                </table>
            </div>
        </div>
        
        <div class="mb-4">
            <h3>Evidence Annotations</h3>
            ${Object.entries(data.annotations.reduce((acc, ann) => {
                if (!acc[ann.concept]) acc[ann.concept] = [];
                acc[ann.concept].push(ann);
                return acc;
            }, {})).map(([concept, annotations]) => `
                <div class="card mb-3">
                    <div class="card-header">
                        <h5 class="mb-0" title="${concept}">${this.cleanConceptText(concept)}</h5>
                    </div>
                    <div class="card-body">
                        ${annotations.map(ann => `
                            <div class="mb-2">
                                <small class="text-muted">${ann.source === 'note' ? 'Note' : 'Generated Summary'}:</small>
                                <blockquote class="blockquote-footer mt-1">
                                    <span class="concept-${ann.colorIndex}">"${ann.text}"</span>
                                </blockquote>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('')}
        </div>
        
        ${data.generatedSummary ? `
        <div class="mb-4">
            <h3>Generated Summary</h3>
            <div class="card">
                <div class="card-body summary-text">${this.markdownToHtml(data.generatedSummary)}</div>
            </div>
        </div>
        ` : ''}
        
        ${data.includeNoteContent ? `
        <div class="mb-4">
            <h3>${data.config.dataset.note_column.charAt(0).toUpperCase() + data.config.dataset.note_column.slice(1)}</h3>
            <div class="card">
                <div class="card-body note-text">${data.selectedNote}</div>
            </div>
        </div>
        ` : ''}
    </div>
</body>
</html>
        `;
    }
    
    markdownToHtml(markdown) {
        if (typeof marked !== 'undefined') {
            return marked.parse(markdown);
        } else {
            return markdown
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/\n\n/g, '</p><p>')
                .replace(/\n/g, '<br>')
                .replace(/^/, '<p>')
                .replace(/$/, '</p>');
        }
    }
    
    showError(elementId, message) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `<div class="error">${message}</div>`;
        }
    }
}

// Initialize the application when the page loads
let app;
document.addEventListener('DOMContentLoaded', function() {
    app = new GenericConceptAnalysisInterface();
});