// Apply the saved theme immediately, before the page paints, to avoid a flash
(function () {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") {
        document.documentElement.setAttribute("data-theme", saved);
    }
})();

document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("themeToggle");
    if (!btn) return;

    function updateIcon() {
        const current = document.documentElement.getAttribute("data-theme");
        btn.textContent = current === "light" ? "☀️" : "🌙";
    }

    btn.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme");
        const next = current === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("theme", next);
        updateIcon();
    });

    updateIcon();
});