(() => {
  "use strict";

  const containers = Array.from(document.querySelectorAll("[data-publications-list]"));
  if (containers.length === 0) {
    return;
  }

  const source = containers[0].dataset.publicationsSource || "publications.json";

  function doiUrl(doi) {
    if (!doi) {
      return "";
    }
    const value = String(doi).trim();
    if (/^https?:\/\//i.test(value)) {
      return value;
    }
    return value.startsWith("10.") ? `https://doi.org/${value}` : value;
  }

  function arxivUrl(arxiv) {
    return arxiv ? `https://arxiv.org/abs/${String(arxiv).trim()}` : "";
  }

  function halUrl(hal) {
    if (!hal) {
      return "";
    }
    const value = String(hal).trim();
    if (/^https?:\/\//i.test(value)) {
      return value;
    }
    return `https://hal.science/hal-${value.replace(/^hal-/, "")}`;
  }

  function primaryUrl(publication) {
    return doiUrl(publication.doi) || arxivUrl(publication.arxiv) || halUrl(publication.hal);
  }

  function publicationTags(publication) {
    const values = [];
    [publication.tags, publication.tag].forEach((field) => {
      if (Array.isArray(field)) {
        values.push(...field);
      } else if (field) {
        values.push(field);
      }
    });

    const unique = new Map();
    values.forEach((tag) => {
      const value = String(tag).trim();
      if (value) {
        unique.set(value.toLocaleLowerCase(), value);
      }
    });
    return Array.from(unique.values());
  }

  function tagLabel(tag) {
    return tag.charAt(0).toLocaleUpperCase() + tag.slice(1);
  }

  function appendLink(container, label, url) {
    if (!url) {
      return;
    }
    if (container.childNodes.length > 0) {
      container.append(" · ");
    }
    const link = document.createElement("a");
    link.href = url;
    link.textContent = label;
    container.append(link);
  }

  function typesetMath(element) {
    if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Queue) {
      window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub, element]);
    }
  }

  function createAbstract(publication, id) {
    const wrapper = document.createElement("div");
    wrapper.className = "publication-abstract-wrapper";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "publication-toggle";
    button.id = `${id}-toggle`;
    button.setAttribute("aria-controls", id);
    button.setAttribute("aria-expanded", "false");
    button.textContent = "Show abstract";

    const abstract = document.createElement("div");
    abstract.className = "publication-abstract";
    abstract.id = id;
    abstract.setAttribute("role", "region");
    abstract.setAttribute("aria-labelledby", button.id);

    let populated = false;
    button.addEventListener("click", () => {
      const opening = !abstract.classList.contains("is-open");
      if (opening && !populated) {
        const text = document.createElement("p");
        text.textContent = publication.abstract;
        abstract.append(text);
        populated = true;
        typesetMath(abstract);
      }
      abstract.classList.toggle("is-open", opening);
      button.setAttribute("aria-expanded", String(opening));
      button.textContent = opening ? "Hide abstract" : "Show abstract";
    });

    wrapper.append(button, abstract);
    return wrapper;
  }

  function createPublication(publication, options) {
    const card = document.createElement("article");
    card.className = options.latest
      ? "ex publication-card publication-card-latest"
      : "pubitemnew publication-card";

    const titleLine = document.createElement("div");
    titleLine.className = "publication-title";
    if (options.number) {
      const number = document.createElement("span");
      number.className = "publication-number";
      number.textContent = `[${options.number}] `;
      titleLine.append(number);
    }
    const title = document.createElement("strong");
    const url = primaryUrl(publication);
    if (options.latest && url) {
      const link = document.createElement("a");
      link.href = url;
      link.textContent = publication.title;
      title.append(link);
    } else {
      title.textContent = publication.title;
    }
    titleLine.append(title);

    const authors = document.createElement("div");
    authors.className = "publication-authors";
    const authorText = document.createElement("em");
    authorText.textContent = publication.author;
    authors.append(authorText);

    const metadata = document.createElement("div");
    metadata.className = "publication-metadata";
    metadata.textContent = [
      publication.journal,
      options.showYear === false ? null : publication.year,
    ].filter(Boolean).join(", ");

    const links = document.createElement("div");
    links.className = "publication-links";
    appendLink(links, "Journal", doiUrl(publication.doi));
    appendLink(links, `arXiv:${publication.arxiv}`, arxivUrl(publication.arxiv));
    appendLink(links, `HAL:${publication.hal}`, halUrl(publication.hal));

    card.append(titleLine, authors, metadata);
    if (links.childNodes.length > 0) {
      card.append(links);
    }
    if (options.showTags) {
      const tags = publicationTags(publication);
      if (tags.length > 0) {
        const tagList = document.createElement("div");
        tagList.className = "publication-tags";
        tagList.setAttribute("aria-label", "Topics");
        tags.forEach((tag) => {
          const badge = document.createElement("span");
          badge.className = "publication-tag";
          badge.textContent = tag;
          tagList.append(badge);
        });
        card.append(tagList);
      }
    }
    if (publication.abstract) {
      card.append(createAbstract(publication, options.abstractId));
    }
    return card;
  }

  function renderLatest(container, data, containerIndex) {
    const limit = Number.parseInt(container.dataset.publicationsLimit || "5", 10);
    const publications = data.slice(-limit).reverse();
    const fragment = document.createDocumentFragment();

    publications.forEach((publication, index) => {
      fragment.append(
        createPublication(publication, {
          latest: true,
          number: null,
          abstractId: `publication-abstract-${containerIndex}-${index}`,
          showTags: false,
          showYear: true,
        })
      );
    });
    container.replaceChildren(fragment);
  }

  function renderFullList(container, data, containerIndex) {
    const records = data.map((publication, index) => ({
      publication,
      index,
      tags: publicationTags(publication),
    }));
    const tagCounts = new Map();
    records.forEach((record) => {
      record.tags.forEach((tag) => {
        tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
      });
    });
    const availableTags = Array.from(tagCounts.keys()).sort((a, b) => a.localeCompare(b));
    const activeTags = new Set();

    const controls = document.createElement("div");
    controls.className = "publication-filters";

    const label = document.createElement("p");
    label.className = "publication-filter-label";
    label.textContent = "Filter by topic:";

    const buttonGroup = document.createElement("div");
    buttonGroup.className = "publication-filter-buttons";
    buttonGroup.setAttribute("role", "group");
    buttonGroup.setAttribute("aria-label", "Filter publications by topic");

    const allButton = document.createElement("button");
    allButton.type = "button";
    allButton.className = "but3 publication-filter-button";
    allButton.textContent = `All publications (${data.length})`;
    allButton.setAttribute("aria-pressed", "true");
    buttonGroup.append(allButton);

    const tagButtons = new Map();
    availableTags.forEach((tag) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "but3 publication-filter-button";
      button.textContent = `${tagLabel(tag)} (${tagCounts.get(tag)})`;
      button.setAttribute("aria-pressed", "false");
      tagButtons.set(tag, button);
      buttonGroup.append(button);
    });

    const summary = document.createElement("p");
    summary.className = "publication-filter-summary";
    summary.setAttribute("aria-live", "polite");

    const results = document.createElement("div");
    results.className = "publication-year-groups";

    controls.append(label, buttonGroup, summary);
    container.replaceChildren(controls, results);

    function renderResults() {
      const filtered = activeTags.size === 0
        ? records
        : records.filter((record) =>
          Array.from(activeTags).every((tag) => record.tags.includes(tag))
        );
      const groups = new Map();
      filtered.forEach((record) => {
        const year = String(record.publication.year || "Other");
        if (!groups.has(year)) {
          groups.set(year, []);
        }
        groups.get(year).push(record);
      });

      const years = Array.from(groups.keys()).sort((a, b) => {
        const numericDifference = Number(b) - Number(a);
        return Number.isNaN(numericDifference) ? b.localeCompare(a) : numericDifference;
      });
      const fragment = document.createDocumentFragment();

      years.forEach((year) => {
        const section = document.createElement("section");
        section.className = "publication-year-section";

        const heading = document.createElement("h4");
        heading.className = "publication-year";
        heading.id = `publication-year-${containerIndex}-${year.replace(/[^a-z0-9]+/gi, "-")}`;
        const yearText = document.createElement("span");
        yearText.textContent = year;
        heading.append(yearText);
        section.setAttribute("aria-labelledby", heading.id);
        section.append(heading);

        groups.get(year).slice().reverse().forEach((record) => {
          section.append(
            createPublication(record.publication, {
              latest: false,
              number: record.index + 1,
              abstractId: `publication-abstract-${containerIndex}-${record.index}`,
              showTags: true,
              showYear: false,
            })
          );
        });
        fragment.append(section);
      });

      if (filtered.length === 0) {
        const empty = document.createElement("p");
        empty.className = "publication-status";
        empty.textContent = "No publications match the selected topics.";
        fragment.append(empty);
      }

      results.replaceChildren(fragment);
      summary.textContent = activeTags.size === 0
        ? `Showing all ${data.length} publications.`
        : `Showing ${filtered.length} of ${data.length} publications.`;
      allButton.setAttribute("aria-pressed", String(activeTags.size === 0));
      tagButtons.forEach((button, tag) => {
        button.setAttribute("aria-pressed", String(activeTags.has(tag)));
      });
    }

    allButton.addEventListener("click", () => {
      activeTags.clear();
      renderResults();
    });
    tagButtons.forEach((button, tag) => {
      button.addEventListener("click", () => {
        if (activeTags.has(tag)) {
          activeTags.delete(tag);
        } else {
          activeTags.add(tag);
        }
        renderResults();
      });
    });

    renderResults();
  }

  function renderContainer(container, data, containerIndex) {
    if (container.dataset.publicationsList === "latest") {
      renderLatest(container, data, containerIndex);
    } else {
      renderFullList(container, data, containerIndex);
    }
  }

  fetch(source)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      if (!Array.isArray(data)) {
        throw new TypeError("The publication data is not an array");
      }
      containers.forEach((container, index) => renderContainer(container, data, index));
    })
    .catch((error) => {
      console.error("Unable to load publications:", error);
      containers.forEach((container) => {
        const status = document.createElement("p");
        status.className = "publication-status publication-status-error";
        status.textContent = "The publication list could not be loaded. Please refresh the page.";
        container.replaceChildren(status);
      });
    });
})();
