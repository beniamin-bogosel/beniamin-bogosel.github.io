(() => {
  "use strict";

  const containers = Array.from(document.querySelectorAll("[data-talks-list]"));
  if (containers.length === 0) {
    return;
  }

  const source = containers[0].dataset.talksSource || "talks.json";
  const months = new Map([
    ["january", 1],
    ["february", 2],
    ["march", 3],
    ["april", 4],
    ["may", 5],
    ["june", 6],
    ["july", 7],
    ["august", 8],
    ["september", 9],
    ["october", 10],
    ["november", 11],
    ["december", 12],
  ]);

  function dateParts(date) {
    const value = String(date || "").trim();
    const yearMatch = value.match(/(?:19|20)\d{2}/);
    if (!yearMatch) {
      return null;
    }
    const lowerDate = value.toLocaleLowerCase();
    const monthEntry = Array.from(months).find(([month]) => lowerDate.includes(month));
    return {
      year: Number.parseInt(yearMatch[0], 10),
      month: monthEntry ? monthEntry[1] : 0,
    };
  }

  function dateValue(date) {
    const parts = dateParts(date);
    return parts ? parts.year * 100 + parts.month : Number.NEGATIVE_INFINITY;
  }

  function machineDate(date) {
    const parts = dateParts(date);
    if (!parts) {
      return "";
    }
    return parts.month
      ? `${parts.year}-${String(parts.month).padStart(2, "0")}`
      : String(parts.year);
  }

  function appendSeparator(container) {
    if (container.childNodes.length > 0) {
      container.append(", ");
    }
  }

  function appendEvent(container, talk) {
    if (!talk.event) {
      return;
    }
    appendSeparator(container);
    if (talk.event_url) {
      const link = document.createElement("a");
      link.href = talk.event_url;
      link.textContent = talk.event;
      container.append(link);
    } else {
      container.append(talk.event);
    }
  }

  function appendPlace(container, place) {
    if (!place) {
      return;
    }
    appendSeparator(container);
    container.append(place);
  }

  function appendDate(container, date) {
    if (!date) {
      return;
    }
    appendSeparator(container);
    const time = document.createElement("time");
    const dateTime = machineDate(date);
    if (dateTime) {
      time.dateTime = dateTime;
    }
    time.textContent = date;
    container.append(time);
  }

  function createTalk(talk) {
    const item = document.createElement("li");
    item.className = "talk-item";

    if (talk.title) {
      const title = document.createElement("strong");
      title.textContent = talk.title;
      item.append(title);
    }

    const details = document.createElement("span");
    appendEvent(details, talk);
    appendPlace(details, talk.place);
    appendDate(details, talk.date);
    if (details.childNodes.length > 0) {
      if (talk.title) {
        item.append(" — ");
      }
      item.append(details);
    }

    if (talk.slides) {
      if (item.childNodes.length > 0) {
        item.append(" · ");
      }
      const slides = document.createElement("a");
      slides.href = talk.slides;
      slides.textContent = "slides";
      item.append(slides);
    }

    return item;
  }

  function sortedTalks(data) {
    return data
      .map((talk, index) => ({ talk, index, date: dateValue(talk.date) }))
      .sort((a, b) => b.date - a.date || a.index - b.index)
      .map((record) => record.talk);
  }

  function renderContainer(container, data) {
    const limit = container.dataset.talksList === "latest"
      ? Number.parseInt(container.dataset.talksLimit || "5", 10)
      : data.length;
    const list = document.createElement("ul");
    list.className = "talk-list";
    sortedTalks(data).slice(0, limit).forEach((talk) => list.append(createTalk(talk)));
    container.replaceChildren(list);
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
        throw new TypeError("The talks data is not an array");
      }
      containers.forEach((container) => renderContainer(container, data));
    })
    .catch((error) => {
      console.error("Unable to load talks:", error);
      containers.forEach((container) => {
        const status = document.createElement("p");
        status.className = "talk-status talk-status-error";
        status.textContent = "The talks list could not be loaded. Please refresh the page.";
        container.replaceChildren(status);
      });
    });
})();
