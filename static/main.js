document.addEventListener('DOMContentLoaded', function() {
    // --- DOM Elements ---
    const container = document.getElementById('visualization');
    const graphContainer = document.getElementById('graph-container');
    const quizContainer = document.getElementById('quiz-container');
    const quizForm = document.getElementById('quiz-form');
    const startQuizBtn = document.getElementById('start-quiz-btn');
    const submitQuizBtn = document.getElementById('submit-quiz-btn');
    const backToGraphBtn = document.getElementById('confirm-and-back-to-graph-btn');
    const nodeTagHighlighterList = document.getElementById('node-tag-highlighter-list');
    const edgeTagHighlighterList = document.getElementById('edge-tag-highlighter-list');
    const llmQueryBtn = document.getElementById('llm-query-btn');
    const llmQueryInput = document.getElementById('llm-query-input');
    const llmOutput = document.getElementById('llm-output');
    const nodeDetailsPanel = document.getElementById('node-details-panel');
    const nodeDetailsTitle = document.getElementById('node-details-title');
    const nodeTagsInput = document.getElementById('node-tags-input');
    const nodeDescriptionInput = document.getElementById('node-description-input');
    const suggestionsPanel = document.getElementById('suggestions-panel');
    const suggestionsTitle = document.getElementById('suggestions-title');
    const suggestionsListDiv = document.getElementById('suggestions-list');
    const edgeEditorModal = document.getElementById('edge-editor-modal');
    const edgeEditorTitle = document.getElementById('edge-editor-title');
    const edgeEditorLabelInput = document.getElementById('edge-editor-label-input');
    const edgeEditorTagsInput = document.getElementById('edge-editor-tags-input');
    const edgeEditorDescriptionInput = document.getElementById('edge-editor-description-input');
    const searchInput = document.getElementById('search-input');
    const nodeIdInput = document.getElementById('node-id-input');
    const saveSnapshotBtn = document.getElementById('save-snapshot-btn');
    const loadSnapshotBtn = document.getElementById('load-snapshot-btn');
    const snapshotList = document.getElementById('snapshot-list');

    let network = null;
    let rawGraphData = { nodes: [], links: [] };
    let visData = { nodes: [], edges: [] };
    let currentFocusNode = null;
    let currentEditingEdge = null;
    let currentQuizData = [];

    // --- Options for vis.js ---
    const overviewOptions = { nodes: { shape: 'dot', size: 10 }, edges: { width: 0.5, color: { inherit: 'from' }, smooth: { type: 'continuous' } }, physics: { enabled: true, forceAtlas2Based: { gravitationalConstant: -26, centralGravity: 0.005, springLength: 230, springConstant: 0.18 }, maxVelocity: 146, solver: 'forceAtlas2Based', timestep: 0.35, stabilization: { iterations: 150 } }, interaction: { tooltipDelay: 200, hideEdgesOnDrag: true } };
    const zoomOptions = { nodes: { shape: 'box', font: { size: 16 } }, edges: { font: { align: 'middle' } }, physics: { enabled: true, solver: 'barnesHut' }, interaction: { navigationButtons: true, keyboard: true } };

    // --- Helper & Drawing Functions ---
    function htmlToElement(html) { const template = document.createElement('div'); template.innerHTML = html.trim(); return template; }

    function drawGraph(dataToDraw, options) {
        const dataForVis = { nodes: [], edges: [] };
        dataForVis.nodes = dataToDraw.nodes.map(n => ({...n, title: n.title ? htmlToElement(n.title) : undefined }));
        dataForVis.edges = dataToDraw.edges.map(e => ({...e, title: e.title ? htmlToElement(e.title) : undefined }));
        if (network) network.destroy();
        network = new vis.Network(container, dataForVis, options);
        setupEventListeners();
    }

    function applyHighlights(visDataToHighlight) {
        const highlightedNodeTags = Array.from(nodeTagHighlighterList.querySelectorAll('input:checked')).map(cb => cb.value);
        const highlightedEdgeTags = Array.from(edgeTagHighlighterList.querySelectorAll('input:checked')).map(cb => cb.value);
        if (highlightedNodeTags.length === 0 && highlightedEdgeTags.length === 0) {
            return visDataToHighlight;
        }

        const dataWithHighlights = JSON.parse(JSON.stringify(visDataToHighlight));
        const highlightedNodeIds = new Set();
        
        if (highlightedNodeTags.length > 0) {
            dataWithHighlights.nodes.forEach(node => {
                const rawNode = rawGraphData.nodes.find(n => n.id === node.id);
                const nodeTags = rawNode?.title?.tags || [];
                if (nodeTags.some(tag => highlightedNodeTags.includes(tag))) {
                    node.color = { border: '#e67e22', background: '#f5cba7' };
                    node.borderWidth = 3;
                    highlightedNodeIds.add(node.id);
                }
            });
        }

        dataWithHighlights.edges.forEach(edge => {
            const isConnectedToHighlightedNode = highlightedNodeIds.has(edge.from) || highlightedNodeIds.has(edge.to);
            const rawEdge = rawGraphData.links.find(e => e.source === edge.from && e.target === edge.to);
            const edgeTags = rawEdge?.details?.tags || [];
            const hasHighlightTag = highlightedEdgeTags.length > 0 && edgeTags.some(tag => highlightedEdgeTags.includes(tag));
            
            if (hasHighlightTag) {
                edge.color = { color: '#e67e22', highlight: '#f5cba7' };
                edge.width = 3;
            } else if (isConnectedToHighlightedNode) {
                edge.color = { inherit: false, color: '#2b7ce9' };
            }
        });

        return dataWithHighlights;
    }

    async function postActionRefresh(saveSnapshot = false) {
        if (saveSnapshot) {
            await fetch('/api/graph/save_snapshot', { method: 'POST' });
        }

        // 1. Re-fetch all graph data to keep local copies in sync
        const response = await fetch('/api/graph/full');
        const data = await response.json();
        rawGraphData = data.raw_data;
        visData = data.vis_data;

        // 2. Re-fetch tags as they might have changed
        await fetchAndDisplayTags('nodes', nodeTagHighlighterList);
        await fetchAndDisplayTags('edges', edgeTagHighlighterList);
        await loadNodeList(); // Also refresh the searchable node list
        await fetchAndDisplaySnapshots(); // And the snapshot list

        // 3. Re-draw the correct view (overview or zoomed-in)
        await refreshCurrentView();
    }
    
    async function refreshCurrentView() {
        if (currentFocusNode) {
            await drawZoomedInGraph(currentFocusNode);
        } else {
            const highlightedVisData = applyHighlights(visData);
            drawGraph(highlightedVisData, overviewOptions);
        }
    }

    async function fullAppReload() {
        await drawOverviewGraph();
        await loadNodeList();
        await fetchAndDisplayTags('nodes', nodeTagHighlighterList);
        await fetchAndDisplayTags('edges', edgeTagHighlighterList);
        await fetchAndDisplaySnapshots();
    }

    async function drawOverviewGraph() {
        currentFocusNode = null;
        nodeDetailsPanel.style.display = 'none';
        suggestionsPanel.style.display = 'none';
        const response = await fetch('/api/graph/full');
        const data = await response.json();
        rawGraphData = data.raw_data;
        visData = data.vis_data;
        const highlightedVisData = applyHighlights(visData);
        drawGraph(highlightedVisData, overviewOptions);
    }
    
    async function drawZoomedInGraph(nodeId) {
        if (!nodeId) return;
        currentFocusNode = nodeId;
        nodeIdInput.value = nodeId;
        const response = await fetch(`/api/graph/node/${nodeId}`);
        if (!response.ok) { alert(`Error: Could not find node "${nodeId}"`); currentFocusNode = null; return; }
        const data = await response.json();
        const highlightedVisData = applyHighlights(data.vis_data);
        drawGraph(highlightedVisData, zoomOptions);
        
        const focusNodeRawData = rawGraphData.nodes.find(n => n.id === nodeId);
        populateAndShowEditPanels(nodeId, focusNodeRawData?.title || {});
    }

    function populateAndShowEditPanels(nodeId, details) {
        nodeDetailsTitle.innerText = `for "${nodeId}"`;
        nodeTagsInput.value = (details.tags || []).join(', ');
        nodeDescriptionInput.value = details.description || '';
        nodeDetailsPanel.style.display = 'block';
        suggestionsTitle.innerText = `for "${nodeId}"`;
        suggestionsPanel.style.display = 'block';
        suggestionsListDiv.innerHTML = '<p>Click "Suggest New Nodes" to begin.</p>';
    }

    async function loadNodeList() {
        const response = await fetch('/api/graph/nodes/list');
        const nodes = await response.json();
        const datalist = document.getElementById('nodelist');
        datalist.innerHTML = '';
        nodes.forEach(node => { const option = document.createElement('option'); option.value = node; datalist.appendChild(option); });
    }

    async function fetchAndDisplayTags(type, listElement) {
        const response = await fetch(`/api/graph/tags/${type}`);
        const tags = await response.json();
        listElement.innerHTML = '';
        if (tags.length === 0) { listElement.innerHTML = 'No tags found.'; return; }
        tags.forEach(tag => {
            const label = document.createElement('label');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = tag;
            checkbox.addEventListener('change', refreshCurrentView);
            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(` ${tag}`));
            listElement.appendChild(label);
        });
    }

    async function fetchAndDisplaySnapshots() {
        const response = await fetch('/api/graph/snapshots/list');
        const files = await response.json();
        snapshotList.innerHTML = '';
        if (files.length === 0) {
            snapshotList.innerHTML = '<option>No snapshots saved yet.</option>';
        } else {
            files.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file;
                snapshotList.appendChild(option);
            });
        }
    }

    // --- Event Listeners ---
    function setupEventListeners() {
        // FIXED: Changed from an arrow function to a traditional `function` declaration.
        // This is crucial because `vis.js` binds `this` to the network instance,
        // which is needed to access `this.body.data.edges`. Arrow functions do not
        // have their own `this` context, which was causing the error.
        network.on("click", async function (params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                fetch(`/api/graph/node/${nodeId}/increment_click`, { method: 'POST' });
                const node1Input = document.getElementById('edge-node1-input');
                const node2Input = document.getElementById('edge-node2-input');
                if (node1Input.value === '') { node1Input.value = nodeId; } 
                else if (node2Input.value === '') { node2Input.value = nodeId; }
                drawZoomedInGraph(nodeId);
            } else if (params.edges.length > 0) {
                const edge = this.body.data.edges.get(params.edges[0]);
                currentEditingEdge = { from: edge.from, to: edge.to };
                const response = await fetch(`/api/graph/edge?node1=${edge.from}&node2=${edge.to}`);
                const data = await response.json();
                if (response.ok) {
                    document.getElementById('edge-editor-title').innerText = `Edit Edge: ${edge.from} ↔ ${edge.to}`;
                    document.getElementById('edge-editor-label-input').value = data.label || '';
                    document.getElementById('edge-editor-tags-input').value = (data.details.tags || []).join(', ');
                    document.getElementById('edge-editor-description-input').value = data.details.description || '';
                    document.getElementById('edge-editor-modal').style.display = 'flex';
                } else { alert(`Error: ${data.error}`); }
            }
        });
    }

    // --- Quiz Mode Logic ---
    async function startQuiz() {
        quizForm.innerHTML = '<p>Generating your quiz, please wait...</p>';
        graphContainer.style.display = 'none';
        quizContainer.style.display = 'block';
        const response = await fetch('/api/quiz/generate');
        const questions = await response.json();
        if (!response.ok) {
            quizForm.innerHTML = `<p style="color: red;">Error: ${questions.error}</p>`;
            submitQuizBtn.style.display = 'none';
            backToGraphBtn.style.display = 'block';
            return;
        }
        currentQuizData = questions;
        quizForm.innerHTML = '';
        questions.forEach((q, index) => {
            const questionDiv = document.createElement('div');
            questionDiv.className = 'quiz-question';
            questionDiv.id = `question-${index}`;
            questionDiv.dataset.answer = q.answer;
            questionDiv.dataset.type = q.type;
            let inputHtml = '';
            if (q.type === 'mcq') {
                inputHtml = q.options.map(option => `<label><input type="radio" name="q${index}" value="${option}"> ${option}</label>`).join('');
            } else {
                inputHtml = `<input type="text" name="q${index}" class="fill-in-blank-input" placeholder="Your answer here...">`;
            }
            questionDiv.innerHTML = `
                <div class="question-header">
                    <p>${index + 1}. ${q.question}</p>
                    <label class="flag-question" style="display: none;">
                        Flag <input type="checkbox" class="flag-checkbox">
                    </label>
                </div>
                <div class="quiz-options">${inputHtml}</div>
                <div class="feedback" style="display: none;"></div>`;
            quizForm.appendChild(questionDiv);
        });
        submitQuizBtn.style.display = 'block';
        backToGraphBtn.style.display = 'none';
        document.getElementById('quiz-header').innerText = 'Knowledge Quiz';
    }

    function submitQuiz() {
        currentQuizData.forEach((q, index) => {
            const questionDiv = document.getElementById(`question-${index}`);
            let userAnswer, isCorrect = false;
            if (q.type === 'mcq') {
                const selectedOption = quizForm.querySelector(`input[name="q${index}"]:checked`);
                userAnswer = selectedOption ? selectedOption.value : null;
                isCorrect = userAnswer === q.answer;
            } else {
                const inputElem = quizForm.querySelector(`input[name="q${index}"]`);
                userAnswer = inputElem.value;
                isCorrect = userAnswer.trim().toLowerCase() === q.answer.trim().toLowerCase();
                inputElem.disabled = true;
            }
            const feedbackDiv = questionDiv.querySelector('.feedback');
            feedbackDiv.style.display = 'block';
            questionDiv.querySelector('.flag-question').style.display = 'inline-flex';
            if (isCorrect) {
                questionDiv.classList.add('correct');
                feedbackDiv.innerText = 'Correct!';
            } else {
                questionDiv.classList.add('incorrect');
                feedbackDiv.innerText = `Incorrect. The correct answer is: ${q.answer}`;
            }
        });
        quizForm.querySelectorAll('input[type="radio"]').forEach(input => input.disabled = true);
        document.getElementById('quiz-header').innerText = `Quiz Complete! Review your answers.`;
        submitQuizBtn.style.display = 'none';
        backToGraphBtn.style.display = 'block';
    }

    function confirmAndReturnToGraph() {
        const resultsToSave = [];
        currentQuizData.forEach((q, index) => {
            const questionDiv = document.getElementById(`question-${index}`);
            let userAnswer;
            if (q.type === 'mcq') {
                userAnswer = quizForm.querySelector(`input[name="q${index}"]:checked`)?.value || null;
            } else {
                userAnswer = quizForm.querySelector(`input[name="q${index}"]`).value;
            }
            const isFlagged = questionDiv.querySelector('.flag-checkbox').checked;
            resultsToSave.push({ ...q, userAnswer, flagged: isFlagged });
        });
        fetch('/api/quiz/save_results', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(resultsToSave) });
        quizContainer.style.display = 'none';
        graphContainer.style.display = 'flex';
    }

    // --- LLM Placeholder ---
    async function queryLLM(queryText) {
        llmOutput.innerText = "Querying AI... Please wait.";
        const response = await fetch('/api/llm/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: queryText})
        });
        const data = await response.json();
        llmOutput.innerText = data.response;
    }

    // --- UI Button Bindings ---
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.addEventListener('click', () => {
            header.classList.toggle('collapsed');
            const content = header.nextElementSibling;
            content.classList.toggle('collapsed');
        });
    });

    startQuizBtn.addEventListener('click', startQuiz);
    submitQuizBtn.addEventListener('click', submitQuiz);
    backToGraphBtn.addEventListener('click', confirmAndReturnToGraph);
    llmQueryBtn.addEventListener('click', () => queryLLM(llmQueryInput.value));
    document.getElementById('overview-btn').addEventListener('click', drawOverviewGraph);
    document.getElementById('search-btn').addEventListener('click', () => drawZoomedInGraph(searchInput.value));
    
    saveSnapshotBtn.addEventListener('click', async () => {
        const response = await fetch('/api/graph/save_snapshot', { method: 'POST' });
        const result = await response.json();
        alert(`Snapshot saved as ${result.filename}`);
        await fetchAndDisplaySnapshots();
    });
    loadSnapshotBtn.addEventListener('click', async () => {
        const filename = snapshotList.value;
        if (!filename || filename.includes('...')) return;
        if (!confirm(`This will discard all current changes and load the snapshot '${filename}'. Are you sure?`)) return;
        const response = await fetch('/api/graph/load_snapshot', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({filename}) });
        if (response.ok) {
            alert(`Snapshot ${filename} loaded successfully.`);
            await fullAppReload();
        } else {
            alert(`Error loading snapshot.`);
        }
    });

    document.getElementById('update-node-details-btn').addEventListener('click', async () => {
        if (!currentFocusNode) return;
        const payload = { tags: nodeTagsInput.value.split(',').map(tag => tag.trim()).filter(Boolean), description: nodeDescriptionInput.value };
        const response = await fetch(`/api/graph/node/${currentFocusNode}/update`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const result = await response.json();
        if (response.ok) {
            // A true value here tells the function to save a snapshot
            await postActionRefresh(true); 
        } else {
            alert(`Error: ${result.error}`);
        }
    });
    
    document.getElementById('save-edge-details-btn').addEventListener('click', async () => {
        if (!currentEditingEdge) return;
        const { from, to } = currentEditingEdge;
        const payload = { label: edgeEditorLabelInput.value, details: { tags: edgeEditorTagsInput.value.split(',').map(tag => tag.trim()).filter(Boolean), description: edgeEditorDescriptionInput.value }};
        const response = await fetch(`/api/graph/edge/update?node1=${from}&node2=${to}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const result = await response.json();
        if (response.ok) {
            edgeEditorModal.style.display = 'none';
            currentEditingEdge = null;
            // A true value here tells the function to save a snapshot
            await postActionRefresh(true);
        } else {
            alert(`Error: ${result.error}`);
        }
    });

    document.getElementById('add-node-btn').addEventListener('click', async () => {
        if (!nodeIdInput.value) return alert('Please enter a node ID.');
        const response = await fetch('/api/graph/add_node', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: nodeIdInput.value }) });
        const result = await response.json();
        if (response.ok) {
            alert(`Node "${result.node_id}" added.`);
            // Use postActionRefresh to preserve the view without saving a snapshot
            await postActionRefresh(false); 
            nodeIdInput.value = '';
        } else {
            alert(`Error: ${result.error}`);
        }
    });

    document.getElementById('remove-node-btn').addEventListener('click', async () => {
        const nodeIdToRemove = nodeIdInput.value;
        if (!nodeIdToRemove) return alert('Please select a node to remove.');
        if (!confirm(`Are you sure you want to remove node "${nodeIdToRemove}"?`)) return;
        await fetch('/api/graph/remove_node', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: nodeIdToRemove }) });
        await fullAppReload();
        nodeIdInput.value = '';
    });

    document.getElementById('add-edge-btn').addEventListener('click', async () => {
        const node1 = document.getElementById('edge-node1-input').value;
        const node2 = document.getElementById('edge-node2-input').value;
        if (!node1 || !node2) return alert('Please provide IDs for both nodes.');
        const response = await fetch('/api/graph/add_edge', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ node1, node2 }) });
        const result = await response.json();
        if (response.ok) { 
            alert(`New edge created. Click it to add details.`);
            // Use postActionRefresh to preserve the view without saving a snapshot
            await postActionRefresh(false);
        } else { 
            alert(`Error: ${result.error}`);
        }
    });

    document.getElementById('remove-edge-btn').addEventListener('click', async () => {
        const node1 = document.getElementById('edge-node1-input').value;
        const node2 = document.getElementById('edge-node2-input').value;
        if (!node1 || !node2) return alert('Please provide IDs for both nodes.');
        if (!confirm(`Remove edge between "${node1}" and "${node2}"?`)) return;
        const response = await fetch('/api/graph/remove_edge', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ node1, node2 }) });
        const result = await response.json();
        if (response.ok) { 
            alert(`Edge removed.`);
            // Use postActionRefresh to preserve the view without saving a snapshot
            await postActionRefresh(false); 
        } else { 
            alert(`Error: ${result.error}`);
        }
    });

    document.getElementById('suggest-nodes-btn').addEventListener('click', async () => {
        if (!currentFocusNode) return alert('Please zoom into a node first to get suggestions.');
        suggestionsListDiv.innerHTML = '<p>Getting AI suggestions...</p>';
        const response = await fetch('/api/llm/suggest_nodes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ node_id: currentFocusNode })
        });
        const suggestions = await response.json();
        suggestionsListDiv.innerHTML = '';
        if (!Array.isArray(suggestions) || suggestions.length === 0) { suggestionsListDiv.innerHTML = '<p>Could not get suggestions.</p>'; return; }
        suggestions.forEach(name => {
            const label = document.createElement('label');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = name;
            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(` ${name}`));
            suggestionsListDiv.appendChild(label);
        });
    });

    document.getElementById('add-selected-nodes-btn').addEventListener('click', async () => {
        if (!currentFocusNode) return alert('Please zoom into a node first.');
        const selectedNodes = Array.from(suggestionsListDiv.querySelectorAll('input:checked')).map(cb => cb.value);
        if (selectedNodes.length === 0) return alert('Please select at least one node to add.');
        const response = await fetch('/api/graph/batch_add', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source_node: currentFocusNode, nodes_to_add: selectedNodes }) });
        const result = await response.json();
        alert(result.message || result.error);
        if (response.ok) {
            // Use postActionRefresh to preserve the view without saving a snapshot
            await postActionRefresh(false);
        }
    });

    // --- INITIALIZATION ---
    fullAppReload();
});