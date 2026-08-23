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
    metadata.textContent = [publication.journal, publication.year].filter(Boolean).join(", ");

    const links = document.createElement("div");
    links.className = "publication-links";
    appendLink(links, "Journal", doiUrl(publication.doi));
    appendLink(links, `arXiv:${publication.arxiv}`, arxivUrl(publication.arxiv));
    appendLink(links, `HAL:${publication.hal}`, halUrl(publication.hal));

    card.append(titleLine, authors, metadata);
    if (links.childNodes.length > 0) {
      card.append(links);
    }
    if (publication.abstract) {
      card.append(createAbstract(publication, options.abstractId));
    }
    return card;
  }

  function renderContainer(container, data, containerIndex) {
    const mode = container.dataset.publicationsList;
    const latest = mode === "latest";
    const limit = Number.parseInt(container.dataset.publicationsLimit || "5", 10);
    const publications = latest
      ? data.slice(-limit).reverse()
      : data.slice().reverse();
    const fragment = document.createDocumentFragment();

    publications.forEach((publication, index) => {
      fragment.append(
        createPublication(publication, {
          latest,
          number: latest ? null : data.length - index,
          abstractId: `publication-abstract-${containerIndex}-${index}`,
        })
      );
    });
    container.replaceChildren(fragment);
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
