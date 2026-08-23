(() => {
  "use strict";

  function typesetMath(element) {
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise([element]).catch((error) => {
        console.error("Unable to typeset publication abstract:", error);
      });
    }
  }

  document.querySelectorAll(".publication-toggle").forEach((button) => {
    const abstract = document.getElementById(button.getAttribute("aria-controls"));
    if (!abstract) {
      return;
    }

    button.addEventListener("click", () => {
      const opening = abstract.hidden;
      abstract.hidden = !opening;
      abstract.classList.toggle("is-open", opening);
      button.setAttribute("aria-expanded", String(opening));
      button.textContent = opening ? "Hide abstract" : "Show abstract";
      if (opening) {
        typesetMath(abstract);
      }
    });
  });

  document.querySelectorAll('[data-publications-list="all"]').forEach((container) => {
    const allButton = container.querySelector('[data-publication-filter="all"]');
    const tagButtons = Array.from(
      container.querySelectorAll('[data-publication-filter]:not([data-publication-filter="all"])')
    );
    const cards = Array.from(container.querySelectorAll("[data-publication-card]"));
    const yearSections = Array.from(container.querySelectorAll("[data-publication-year-section]"));
    const summary = container.querySelector(".publication-filter-summary");
    const noResults = container.querySelector(".publication-no-results");
    const activeTags = new Set();

    if (!allButton || !summary) {
      return;
    }

    function cardTags(card) {
      return new Set(
        (card.dataset.publicationTags || "")
          .split("|")
          .map((tag) => tag.trim().toLocaleLowerCase())
          .filter(Boolean)
      );
    }

    const records = cards.map((card) => ({ card, tags: cardTags(card) }));

    function updateResults() {
      let visibleCount = 0;
      records.forEach(({ card, tags }) => {
        const visible = Array.from(activeTags).every((tag) => tags.has(tag));
        card.hidden = !visible;
        if (visible) {
          visibleCount += 1;
        }
      });

      yearSections.forEach((section) => {
        section.hidden = !section.querySelector("[data-publication-card]:not([hidden])");
      });

      allButton.setAttribute("aria-pressed", String(activeTags.size === 0));
      tagButtons.forEach((button) => {
        button.setAttribute(
          "aria-pressed",
          String(activeTags.has(button.dataset.publicationFilter))
        );
      });
      summary.textContent = activeTags.size === 0
        ? `Showing all ${records.length} publications.`
        : `Showing ${visibleCount} of ${records.length} publications.`;
      if (noResults) {
        noResults.hidden = visibleCount !== 0;
      }
    }

    allButton.addEventListener("click", () => {
      activeTags.clear();
      updateResults();
    });

    tagButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const tag = button.dataset.publicationFilter;
        if (activeTags.has(tag)) {
          activeTags.delete(tag);
        } else {
          activeTags.add(tag);
        }
        updateResults();
      });
    });

    updateResults();
  });
})();
