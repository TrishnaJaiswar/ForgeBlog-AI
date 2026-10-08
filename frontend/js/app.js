"use strict";


/* =========================================================
   FORGEBLOG AI FRONTEND
   ========================================================= */


/* ================= STATE ================= */

const state = {

    currentFilename: null,

    currentMarkdown: "",

    currentTitle: "",

    currentPlan: null,

    currentEvidence: [],

    currentImages: [],

    logs: [],

    isGenerating: false,

    currentStep: null

};


/* ================= DOM ================= */

const $ = (selector) => {
    return document.querySelector(selector);
};

const $$ = (selector) => {
    return [...document.querySelectorAll(selector)];
};


/* ================= INITIALIZATION ================= */

document.addEventListener("DOMContentLoaded", () => {

    setDefaultDate();

    bindEvents();

    loadRecentBlogs();

});


/* ================= EVENTS ================= */

function bindEvents() {

    $("#newBlogBtn").addEventListener(
        "click",
        showGenerator
    );


    $("#startWritingBtn").addEventListener(
        "click",
        showGenerator
    );


    $("#generateForm").addEventListener(
        "submit",
        handleGenerate
    );


    $("#retryBtn").addEventListener(
        "click",
        showGenerator
    );


    $("#downloadMarkdownBtn").addEventListener(
        "click",
        downloadMarkdown
    );


    $("#downloadBundleBtn").addEventListener(
        "click",
        downloadBundle
    );


    $$(".tab-btn").forEach((button) => {

        button.addEventListener(
            "click",
            () => switchTab(button.dataset.tab)
        );

    });

}


/* ================= DATE ================= */

function setDefaultDate() {

    const input = $("#asOf");

    if (!input.value) {

        const today = new Date();

        const year = today.getFullYear();

        const month = String(
            today.getMonth() + 1
        ).padStart(2, "0");

        const day = String(
            today.getDate()
        ).padStart(2, "0");

        input.value =
            `${year}-${month}-${day}`;
    }

}


/* ================= VIEW MANAGEMENT ================= */

function hideAllMainPanels() {

    $("#emptyState").classList.add("hidden");

    $("#generatorPanel").classList.add("hidden");

    $("#progressPanel").classList.add("hidden");

    $("#resultPanel").classList.add("hidden");

    $("#errorPanel").classList.add("hidden");

}


function showGenerator() {

    if (state.isGenerating) {
        return;
    }

    hideAllMainPanels();

    $("#generatorPanel").classList.remove(
        "hidden"
    );

    $("#topic").focus();

}


function showEmptyState() {

    hideAllMainPanels();

    $("#emptyState").classList.remove(
        "hidden"
    );

}


/* ================= GENERATION ================= */

async function handleGenerate(event) {

    event.preventDefault();

    if (state.isGenerating) {
        return;
    }

    const topic = $("#topic").value.trim();

    const asOf = $("#asOf").value;

    const recencyDays =
        Number($("#recencyDays").value);


    if (!topic) {

        $("#topic").focus();

        return;
    }


    if (!asOf) {

        $("#asOf").focus();

        return;
    }


    state.isGenerating = true;

    state.logs = [];

    state.currentFilename = null;

    state.currentMarkdown = "";

    state.currentPlan = null;

    state.currentEvidence = [];

    state.currentImages = [];


    setGeneratingUI(true);

    showProgress();

    resetProgress();


    try {

        const response = await fetch(
            "/api/generate",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    topic: topic,

                    as_of: asOf,

                    recency_days:
                        recencyDays

                })
            }
        );


        if (!response.ok) {

            let message =
                `HTTP ${response.status}`;

            try {

                const errorData =
                    await response.json();

                message =
                    errorData.detail ||
                    message;

            } catch (_) {}

            throw new Error(message);
        }


        await consumeSSE(
            response
        );


    } catch (error) {

        console.error(error);

        showError(
            error.message ||
            "Unable to generate blog."
        );

    } finally {

        state.isGenerating = false;

        setGeneratingUI(false);

    }

}


/* ================= SSE ================= */

async function consumeSSE(response) {

    const reader =
        response.body.getReader();

    const decoder =
        new TextDecoder();

    let buffer = "";


    while (true) {

        const {
            value,
            done
        } = await reader.read();


        if (done) {
            break;
        }


        buffer += decoder.decode(
            value,
            {
                stream: true
            }
        );


        const events =
            buffer.split("\n\n");


        buffer =
            events.pop() || "";


        for (const rawEvent of events) {

            processSSEEvent(
                rawEvent
            );

        }

    }


    if (buffer.trim()) {

        processSSEEvent(buffer);

    }

}


/* ================= SSE PROCESSING ================= */

function processSSEEvent(rawEvent) {

    const lines =
        rawEvent.split("\n");


    let eventName =
        "message";

    let data =
        "";


    for (const line of lines) {

        if (
            line.startsWith(
                "event:"
            )
        ) {

            eventName =
                line.slice(6).trim();

        }


        if (
            line.startsWith(
                "data:"
            )
        ) {

            data +=
                line.slice(5).trim();

        }

    }


    if (!data) {
        return;
    }


    let payload;

    try {

        payload =
            JSON.parse(data);

    } catch (_) {

        payload = {
            message: data
        };

    }


    if (
        eventName === "started"
    ) {

        addLog(
            payload.message ||
            "Generation started."
        );

        updateProgress(
            "router",
            10,
            "Initializing LangGraph workflow..."
        );

    }


    else if (
        eventName === "progress"
    ) {

        handleProgress(
            payload
        );

    }


    else if (
        eventName === "complete"
    ) {

        handleComplete(
            payload
        );

    }


    else if (
        eventName === "error"
    ) {

        showError(
            payload.message ||
            "Generation failed."
        );

    }

}


/* ================= PROGRESS ================= */

function handleProgress(payload) {

    const node =
        String(
            payload.node ||
            ""
        ).toLowerCase();


    const message =
        payload.message ||
        `Running ${node}...`;


    addLog(
        message
    );


    let progress = 20;


    if (
        node.includes("router")
    ) {

        progress = 15;

        updateProgress(
            "router",
            progress,
            message
        );

    }

    else if (
        node.includes("research")
    ) {

        progress = 30;

        completeStep("router");

        updateProgress(
            "research",
            progress,
            message
        );

    }

    else if (
        node.includes("orchestrator") ||
        node.includes("planner")
    ) {

        progress = 50;

        completeStep("research");

        updateProgress(
            "orchestrator",
            progress,
            message
        );

    }

    else if (
        node.includes("worker")
    ) {

        progress = 70;

        completeStep("orchestrator");

        updateProgress(
            "worker",
            progress,
            message
        );

    }

    else if (
        node.includes("reducer")
    ) {

        progress = 90;

        completeStep("worker");

        updateProgress(
            "reducer",
            progress,
            message
        );

    }

    else {

        updateProgress(
            null,
            progress,
            message
        );

    }

}


/* ================= PROGRESS UI ================= */

function showProgress() {

    hideAllMainPanels();

    $("#progressPanel").classList.remove(
        "hidden"
    );

}


function resetProgress() {

    state.currentStep = null;

    $("#progressBar").style.width =
        "0%";

    $("#progressPercent").textContent =
        "0%";

    $("#progressMessage").textContent =
        "Initializing workflow...";


    $$(".workflow-step").forEach(
        (step) => {

            step.classList.remove(
                "active",
                "completed"
            );

        }
    );

}


function updateProgress(
    stepName,
    percent,
    message
) {

    $("#progressBar").style.width =
        `${percent}%`;

    $("#progressPercent").textContent =
        `${percent}%`;

    $("#progressMessage").textContent =
        message;


    if (!stepName) {
        return;
    }


    if (
        state.currentStep &&
        state.currentStep !== stepName
    ) {

        completeStep(
            state.currentStep
        );

    }


    const step =
        document.querySelector(
            `.workflow-step[data-step="${stepName}"]`
        );


    if (step) {

        step.classList.add(
            "active"
        );

    }


    state.currentStep =
        stepName;

}


function completeStep(stepName) {

    const step =
        document.querySelector(
            `.workflow-step[data-step="${stepName}"]`
        );


    if (!step) {
        return;
    }


    step.classList.remove(
        "active"
    );

    step.classList.add(
        "completed"
    );


    const icon =
        step.querySelector(
            ".step-icon"
        );


    if (icon) {

        icon.textContent =
            "✓";

    }

}


/* ================= COMPLETE ================= */

function handleComplete(payload) {

    completeStep(
        "reducer"
    );


    $("#progressBar").style.width =
        "100%";

    $("#progressPercent").textContent =
        "100%";

    $("#progressMessage").textContent =
        "Blog generated successfully.";


    addLog(
        "Blog generation completed."
    );


    const result =
        payload.result ||
        payload;


    state.currentMarkdown =
        result.final ||
        result.markdown ||
        "";


    state.currentPlan =
        result.plan ||
        null;


    state.currentEvidence =
        result.evidence ||
        [];


    state.currentImages =
        result.image_specs ||
        result.images ||
        [];


    state.currentTitle =
        getTitle(
            state.currentMarkdown
        ) ||
        result.title ||
        "Generated Blog";


    state.currentFilename =
        result.filename ||
        makeMarkdownFilename(
            state.currentTitle
        );


    renderResult();


    setTimeout(
        () => {

            hideAllMainPanels();

            $("#resultPanel")
                .classList.remove(
                    "hidden"
                );

        },
        400
    );


    loadRecentBlogs();

}


/* ================= RESULT RENDER ================= */

function renderResult() {

    $("#resultTitle").textContent =
        state.currentTitle;


    renderMarkdown();

    renderPlan();

    renderEvidence();

    renderImages();

    renderLogs();

    switchTab(
        "preview"
    );

}


/* ================= MARKDOWN ================= */

function renderMarkdown() {

    const container =
        $("#markdownPreview");


    if (
        !state.currentMarkdown
    ) {

        container.innerHTML =
            "<p>No Markdown was returned.</p>";

        return;

    }


    let markdown =
        state.currentMarkdown;


    markdown =
        resolveMarkdownImages(
            markdown
        );


    if (
        typeof marked !== "undefined"
    ) {

        container.innerHTML =
            marked.parse(
                markdown
            );

    } else {

        container.textContent =
            markdown;

    }

}


/* ================= IMAGE PATHS ================= */

function resolveMarkdownImages(
    markdown
) {

    return markdown.replace(
        /(!\[[^\]]*\]\()([^)]+)(\))/g,
        (
            match,
            prefix,
            path,
            suffix
        ) => {

            const cleanPath =
                path
                    .replace(/^["']|["']$/g, "")
                    .trim();


            if (
                cleanPath.startsWith(
                    "http://"
                ) ||
                cleanPath.startsWith(
                    "https://"
                ) ||
                cleanPath.startsWith(
                    "/api/"
                ) ||
                cleanPath.startsWith(
                    "data:"
                )
            ) {

                return match;

            }


            const filename =
                cleanPath
                    .split("/")
                    .pop();


            return (
                prefix +
                `/api/images/${encodeURIComponent(filename)}` +
                suffix
            );

        }
    );

}


/* ================= PLAN ================= */

function renderPlan() {

    const container =
        $("#planContent");


    const plan =
        state.currentPlan;


    if (!plan) {

        container.innerHTML =
            `
            <div class="empty-tab">
                No plan data available.
            </div>
            `;

        return;

    }


    const tasks =
        Array.isArray(plan.tasks)
            ? plan.tasks
            : [];


    let html = "";


    html += `
        <div class="plan-summary">

            <div class="plan-card">
                <div class="plan-card-label">
                    Blog Title
                </div>

                <div class="plan-card-value">
                    ${escapeHtml(
                        plan.blog_title || "-"
                    )}
                </div>
            </div>

            <div class="plan-card">
                <div class="plan-card-label">
                    Audience
                </div>

                <div class="plan-card-value">
                    ${escapeHtml(
                        plan.audience || "-"
                    )}
                </div>
            </div>

            <div class="plan-card">
                <div class="plan-card-label">
                    Tone
                </div>

                <div class="plan-card-value">
                    ${escapeHtml(
                        plan.tone || "-"
                    )}
                </div>
            </div>

        </div>
    `;


    if (
        plan.constraints &&
        plan.constraints.length
    ) {

        html += `
            <div class="task-card">

                <div class="task-title">
                    Constraints
                </div>

                <div class="task-tags">

                    ${plan.constraints
                        .map(
                            item =>
                                `
                                <span class="task-tag">
                                    ${escapeHtml(item)}
                                </span>
                                `
                        )
                        .join("")
                    }

                </div>

            </div>
        `;

    }


    tasks.forEach(
        (task, index) => {

            const tags =
                Array.isArray(task.tags)
                    ? task.tags
                    : [];


            html += `
                <div class="task-card">

                    <div class="task-header">

                        <div>

                            <div class="task-number">
                                TASK ${String(
                                    task.id ||
                                    index + 1
                                ).padStart(2, "0")}
                            </div>

                            <div class="task-title">
                                ${escapeHtml(
                                    task.title || "-"
                                )}
                            </div>

                        </div>

                        <div class="task-number">
                            ${task.target_words || "-"}
                            words
                        </div>

                    </div>

                    <div class="task-goal">
                        ${escapeHtml(
                            task.goal || ""
                        )}
                    </div>

                    ${
                        task.bullets &&
                        task.bullets.length
                            ? `
                                <ul class="task-goal">
                                    ${task.bullets
                                        .map(
                                            bullet =>
                                                `
                                                <li>
                                                    ${escapeHtml(
                                                        bullet
                                                    )}
                                                </li>
                                                `
                                        )
                                        .join("")
                                    }
                                </ul>
                              `
                            : ""
                    }

                    <div class="task-tags">

                        ${tags
                            .map(
                                tag =>
                                    `
                                    <span class="task-tag">
                                        ${escapeHtml(tag)}
                                    </span>
                                    `
                            )
                            .join("")
                        }

                        ${
                            task.requires_research
                                ? `
                                    <span class="task-tag">
                                        Research
                                    </span>
                                  `
                                : ""
                        }

                        ${
                            task.requires_citations
                                ? `
                                    <span class="task-tag">
                                        Citations
                                    </span>
                                  `
                                : ""
                        }

                        ${
                            task.requires_code
                                ? `
                                    <span class="task-tag">
                                        Code
                                    </span>
                                  `
                                : ""
                        }

                    </div>

                </div>
            `;

        }
    );


    container.innerHTML =
        html;

}


/* ================= EVIDENCE ================= */

function renderEvidence() {

    const container =
        $("#evidenceContent");


    const evidence =
        Array.isArray(
            state.currentEvidence
        )
            ? state.currentEvidence
            : [];


    if (!evidence.length) {

        container.innerHTML =
            `
            <div class="empty-tab">
                No research evidence was collected.
            </div>
            `;

        return;

    }


    const html =
        `
        <div class="evidence-list">

            ${evidence
                .map(
                    item =>
                        `
                        <div class="evidence-card">

                            <div class="evidence-title">
                                ${escapeHtml(
                                    item.title ||
                                    "Untitled source"
                                )}
                            </div>

                            <div class="evidence-meta">

                                ${
                                    escapeHtml(
                                        item.source ||
                                        ""
                                    )
                                }

                                ${
                                    item.published_at
                                        ? " • " +
                                          escapeHtml(
                                              item.published_at
                                          )
                                        : ""
                                }

                            </div>

                            ${
                                item.snippet
                                    ? `
                                        <div class="evidence-snippet">
                                            ${escapeHtml(
                                                item.snippet
                                            )}
                                        </div>
                                      `
                                    : ""
                            }

                            ${
                                item.url
                                    ? `
                                        <a
                                            class="evidence-link"
                                            href="${escapeAttribute(
                                                item.url
                                            )}"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                        >
                                            Open source →
                                        </a>
                                      `
                                    : ""
                            }

                        </div>
                        `
                )
                .join("")
            }

        </div>
        `;


    container.innerHTML =
        html;

}


/* ================= IMAGES ================= */

function renderImages() {

    const container =
        $("#imagesContent");


    const images =
        Array.isArray(
            state.currentImages
        )
            ? state.currentImages
            : [];


    if (!images.length) {

        container.innerHTML =
            `
            <div class="empty-tab">
                No image specifications were generated.
            </div>
            `;

        return;

    }


    const html =
        `
        <div class="image-grid">

            ${images
                .map(
                    image =>
                        `
                        <div class="image-card">

                            ${
                                image.filename
                                    ? `
                                        <img
                                            src="/api/images/${encodeURIComponent(
                                                getFilename(
                                                    image.filename
                                                )
                                            )}"
                                            alt="${escapeAttribute(
                                                image.alt ||
                                                ""
                                            )}"
                                            loading="lazy"
                                            onerror="this.style.display='none'"
                                        >
                                      `
                                    : ""
                            }

                            <div class="image-card-body">

                                <div class="image-card-title">
                                    ${escapeHtml(
                                        image.filename ||
                                        image.placeholder ||
                                        "Generated Image"
                                    )}
                                </div>

                                ${
                                    image.caption
                                        ? `
                                            <div class="image-card-caption">
                                                ${escapeHtml(
                                                    image.caption
                                                )}
                                            </div>
                                          `
                                        : ""
                                }

                            </div>

                        </div>
                        `
                )
                .join("")
            }

        </div>
        `;


    container.innerHTML =
        html;

}


/* ================= LOGS ================= */

function renderLogs() {

    const container =
        $("#logsContent");


    if (!state.logs.length) {

        container.innerHTML =
            `
            <div class="log-line">
                No logs available.
            </div>
            `;

        return;

    }


    container.innerHTML =
        state.logs
            .map(
                log =>
                    `
                    <div class="log-line">
                        ${escapeHtml(log)}
                    </div>
                    `
            )
            .join("");

}


function addLog(message) {

    const timestamp =
        new Date().toLocaleTimeString();


    state.logs.push(
        `[${timestamp}] ${message}`
    );


    if (
        state.logs.length > 200
    ) {

        state.logs =
            state.logs.slice(-200);

    }


    renderLogs();

}


/* ================= TABS ================= */

function switchTab(tabName) {

    $$(".tab-btn").forEach(
        button => {

            button.classList.toggle(
                "active",
                button.dataset.tab === tabName
            );

        }
    );


    $$(".tab-content").forEach(
        content => {

            content.classList.toggle(
                "active",
                content.id ===
                `tab-${tabName}`
            );

        }
    );

}


/* ================= RECENT BLOGS ================= */

async function loadRecentBlogs() {

    const container =
        $("#recentBlogs");


    try {

        const response =
            await fetch(
                "/api/blogs"
            );


        if (!response.ok) {
            throw new Error(
                "Failed to load blogs."
            );
        }


        const data =
            await response.json();


        const blogs =
            Array.isArray(data)
                ? data
                : data.blogs || [];


        if (!blogs.length) {

            container.innerHTML =
                `
                <div class="loading-small">
                    No blogs generated yet.
                </div>
                `;

            return;

        }


        container.innerHTML =
            blogs
                .slice(0, 15)
                .map(
                    blog =>
                        `
                        <button
                            class="recent-blog"
                            type="button"
                            data-filename="${escapeAttribute(
                                blog.filename ||
                                blog.name ||
                                ""
                            )}"
                        >

                            <span
                                class="recent-blog-title"
                            >
                                ${escapeHtml(
                                    blog.title ||
                                    blog.filename ||
                                    "Untitled Blog"
                                )}
                            </span>

                            <span
                                class="recent-blog-date"
                            >
                                ${formatDate(
                                    blog.modified_at ||
                                    blog.created_at ||
                                    ""
                                )}
                            </span>

                        </button>
                        `
                )
                .join("");


        $$(".recent-blog").forEach(
            button => {

                button.addEventListener(
                    "click",
                    () => {

                        loadBlog(
                            button.dataset.filename
                        );

                    }
                );

            }
        );


    } catch (error) {

        console.error(error);

        container.innerHTML =
            `
            <div class="loading-small">
                Unable to load blogs.
            </div>
            `;

    }

}


/* ================= LOAD BLOG ================= */

async function loadBlog(filename) {

    if (!filename) {
        return;
    }


    try {

        hideAllMainPanels();

        $("#progressPanel")
            .classList.remove(
                "hidden"
            );

        $("#progressMessage").textContent =
            "Loading saved blog...";


        const response =
            await fetch(
                `/api/blogs/${encodeURIComponent(filename)}`
            );


        if (!response.ok) {

            throw new Error(
                `Unable to load ${filename}`
            );

        }


        const data =
            await response.json();


        state.currentFilename =
            data.filename ||
            filename;


        state.currentMarkdown =
            data.final ||
            data.markdown ||
            "";


        state.currentPlan =
            data.plan ||
            null;


        state.currentEvidence =
            data.evidence ||
            [];


        state.currentImages =
            data.image_specs ||
            data.images ||
            [];


        state.currentTitle =
            getTitle(
                state.currentMarkdown
            ) ||
            data.title ||
            removeExtension(
                filename
            );


        renderResult();


        hideAllMainPanels();

        $("#resultPanel")
            .classList.remove(
                "hidden"
            );


    } catch (error) {

        console.error(error);

        showError(
            error.message ||
            "Unable to load blog."
        );

    }

}


/* ================= DOWNLOAD ================= */

function downloadMarkdown() {

    if (
        !state.currentFilename
    ) {

        state.currentFilename =
            makeMarkdownFilename(
                state.currentTitle
            );

    }


    window.location.href =
        `/api/download/markdown/${encodeURIComponent(
            state.currentFilename
        )}`;

}


function downloadBundle() {

    if (
        !state.currentFilename
    ) {

        state.currentFilename =
            makeMarkdownFilename(
                state.currentTitle
            );

    }


    window.location.href =
        `/api/download/bundle/${encodeURIComponent(
            state.currentFilename
        )}`;

}


/* ================= UI ================= */

function setGeneratingUI(
    generating
) {

    const button =
        $("#generateBtn");

    const spinner =
        $("#generateSpinner");

    const text =
        $("#generateText");


    button.disabled =
        generating;


    if (generating) {

        spinner.classList.remove(
            "hidden"
        );

        text.textContent =
            "Generating...";

    } else {

        spinner.classList.add(
            "hidden"
        );

        text.textContent =
            "Generate Blog";

    }

}


/* ================= ERROR ================= */

function showError(message) {

    hideAllMainPanels();

    $("#errorPanel")
        .classList.remove(
            "hidden"
        );

    $("#errorMessage").textContent =
        message;


    addLog(
        `ERROR: ${message}`
    );

}


/* ================= HELPERS ================= */

function getTitle(markdown) {

    if (!markdown) {
        return "";
    }


    const match =
        markdown.match(
            /^\s*#\s+(.+)$/m
        );


    if (!match) {
        return "";
    }


    return match[1]
        .replace(
            /\*\*/g,
            ""
        )
        .trim();

}


function makeMarkdownFilename(title) {

    const safe =
        String(title || "forgeblog")
            .toLowerCase()
            .replace(
                /[^a-z0-9]+/g,
                "-"
            )
            .replace(
                /^-+|-+$/g,
                ""
            );


    return `${safe || "forgeblog"}.md`;

}


function removeExtension(filename) {

    return String(filename)
        .replace(
            /\.md$/i,
            ""
        );

}


function getFilename(path) {

    return String(path || "")
        .split("/")
        .pop()
        .split("\\")
        .pop();

}


function escapeHtml(value) {

    return String(value ?? "")
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );

}


function escapeAttribute(value) {

    return escapeHtml(value);

}


function formatDate(value) {

    if (!value) {
        return "";
    }


    const date =
        new Date(value);


    if (
        Number.isNaN(
            date.getTime()
        )
    ) {

        return "";

    }


    return date.toLocaleDateString(
        undefined,
        {
            year: "numeric",
            month: "short",
            day: "numeric"
        }
    );

}