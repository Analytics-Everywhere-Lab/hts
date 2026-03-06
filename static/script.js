document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const productDesc = document.getElementById('product-desc');
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');

    // Accordion Logic
    document.querySelectorAll('.accordion-header').forEach(header => {
        header.addEventListener('click', () => {
            const item = header.parentElement;
            item.classList.toggle('active');
            const content = item.querySelector('.accordion-content');
            if (item.classList.contains('active')) {
                content.style.maxHeight = content.scrollHeight + "px";
            } else {
                content.style.maxHeight = "0";
            }
        });
    });

    generateBtn.addEventListener('click', async () => {
        const desc = productDesc.value.trim();
        if (!desc) return alert("Please enter a product description!");

        // UI Reset
        resultsSection.classList.add('hidden');
        loader.classList.remove('hidden');
        generateBtn.disabled = true;

        const loaderMessage = document.getElementById('loader-message');
        const loadingSteps = [
            "Parsing item description...",
            "Searching DuckDuckGo for context...",
            "Retrieving official Canadian Tariff schedules...",
            "Running self-consistency ensemble 1 of 3...",
            "Running self-consistency ensemble 2 of 3...",
            "Running self-consistency ensemble 3 of 3...",
            "Cross-verifying confidence and vote share...",
            "Finalizing HTS code and extracting reasoning..."
        ];
        let stepIdx = 0;
        loaderMessage.textContent = loadingSteps[0];

        const loaderInterval = setInterval(() => {
            stepIdx++;
            if (stepIdx < loadingSteps.length) {
                loaderMessage.style.opacity = '0';
                setTimeout(() => {
                    loaderMessage.textContent = loadingSteps[stepIdx];
                    loaderMessage.style.transition = 'opacity 0.3s ease';
                    loaderMessage.style.opacity = '1';
                }, 300);
            } else {
                loaderMessage.style.opacity = '0';
                setTimeout(() => {
                    loaderMessage.textContent = "Almost done... rendering final classification.";
                    loaderMessage.style.opacity = '1';
                }, 300);
            }
        }, 3000);

        try {
            const res = await fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description: desc })
            });
            const data = await res.json();

            if (res.ok) {
                renderResults(data);
            } else {
                alert("Error: " + data.detail);
            }
        } catch (err) {
            console.error(err);
            alert("An error occurred while calling the API.");
        } finally {
            clearInterval(loaderInterval);
            loader.classList.add('hidden');
            generateBtn.disabled = false;
        }
    });

    function renderResults(data) {
        resultsSection.classList.remove('hidden');

        // HTS Code & Warning
        document.getElementById('hts-code-display').textContent = data.final_hts_code || "XXXX.XX.XX.XX";

        const warningBox = document.getElementById('escalation-warning');
        if (data.escalation_needed) {
            warningBox.classList.remove('hidden');
        } else {
            warningBox.classList.add('hidden');
        }

        // Confidence Grid
        renderConfidenceGrid(data.element_confidences);

        // Reasoning & Context
        document.getElementById('reasoning-text').innerText = data.reasoning_steps || "No reasoning provided.";
        document.getElementById('search-queries').innerHTML = (data.search_queries || []).map(q => `<li>${q}</li>`).join('');
        document.getElementById('web-context').innerText = data.search_results || "None.";
        document.getElementById('rag-context').innerText = data.rag_context || "None.";

        // Chatbot Initialization
        const chatContainer = document.getElementById('escalation-chat-container');
        const chatMessages = document.getElementById('chat-messages');
        const chatInput = document.getElementById('chat-input');
        const sendChatBtn = document.getElementById('send-chat-btn');
        chatMessages.innerHTML = "";

        const finalCodeContainer = document.getElementById('final-code-container');
        const reasoningPanelContainer = document.getElementById('reasoning-panel-container');

        if (data.escalation_needed) {
            chatContainer.classList.remove('hidden');
            finalCodeContainer.classList.add('hidden');
            reasoningPanelContainer.classList.add('hidden');

            window.currentEscalationContext = {
                description: document.getElementById('product-desc').value.trim(),
                search_results: data.search_results,
                rag_context: data.rag_context,
                chat_history: []
            };

            appendChatMessage("assistant", data.escalation_question || "I need more details to finalize this code.");
        } else {
            chatContainer.classList.add('hidden');
            finalCodeContainer.classList.remove('hidden');
            reasoningPanelContainer.classList.remove('hidden');
        }

        // Chat Handlers
        sendChatBtn.onclick = async () => {
            const text = chatInput.value.trim();
            if (!text) return;

            appendChatMessage("user", text);
            chatInput.value = "";
            sendChatBtn.disabled = true;

            try {
                const res = await fetch('/api/escalation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(window.currentEscalationContext)
                });
                const escData = await res.json();

                if (res.ok && escData.success) {
                    const r = escData.result;
                    if (r.is_final) {
                        appendChatMessage("assistant", "Excellent, I have enough confidence now! Updating the final code.");
                        document.getElementById('hts-code-display').textContent = r.data.final_hts_code;
                        document.getElementById('escalation-warning').classList.add('hidden');
                        document.getElementById('chat-input-area').classList.add('hidden');

                        // Show AI reasoning chain
                        if (r.data.reasoning) {
                            document.getElementById('reasoning-text').innerText = r.data.reasoning;
                        }

                        // Re-render confidence grid with human-verified values
                        if (r.data.element_confidences) {
                            renderConfidenceGrid(r.data.element_confidences);
                        }

                        // Output the final HTS code and reasoning, show the confidence grid
                        document.getElementById('final-code-container').classList.remove('hidden');
                        document.getElementById('reasoning-panel-container').classList.remove('hidden');
                        document.getElementById('confidence-grid').classList.remove('hidden');

                        // Collapse the Chat Accordion
                        const chatAccordionItem = document.getElementById('chat-accordion-item');
                        if (chatAccordionItem) {
                            chatAccordionItem.classList.remove('active');
                            const content = chatAccordionItem.querySelector('.accordion-content');
                            if (content) content.style.maxHeight = "0";
                        }
                    } else {
                        appendChatMessage("assistant", r.message);
                    }
                } else {
                    alert("Chat error: " + (escData.detail || "Unknown error"));
                }
            } catch (err) {
                console.error(err);
                alert("Error sending chat message.");
            } finally {
                sendChatBtn.disabled = false;
            }
        };

        chatInput.onkeypress = (e) => {
            if (e.key === 'Enter') sendChatBtn.click();
        }
    }

    function renderConfidenceGrid(element_confidences) {
        if (!element_confidences) return;

        const grid = document.getElementById('confidence-grid');
        grid.innerHTML = '';
        const order = [
            { key: 'chapter', label: 'Chapter' },
            { key: 'heading', label: 'Heading' },
            { key: 'subheading', label: 'Subheading' },
            { key: 'additional_subheading', label: 'Additional' },
            { key: 'statistical_suffix', label: 'Statistical' }
        ];

        order.forEach(item => {
            const conf = element_confidences[item.key];
            if (!conf) return;

            const isLow = conf.score < 0.6;
            const card = document.createElement('div');
            card.className = `conf-card ${isLow ? 'low-conf' : ''}`;

            card.innerHTML = `
                <div class="label">${item.label}</div>
                <div class="value">${conf.value}</div>
                <div class="score">${conf.confidence}</div>
            `;
            grid.appendChild(card);
        });
    }

    function appendChatMessage(role, text) {
        const chatMessages = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = `chat-msg msg-${role}`;

        if (role === 'assistant' && typeof window.marked !== 'undefined') {
            div.innerHTML = window.marked.parse(text);
        } else {
            div.textContent = text;
        }

        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        if (window.currentEscalationContext) {
            window.currentEscalationContext.chat_history.push({ role, content: text });
        }
    }
});
