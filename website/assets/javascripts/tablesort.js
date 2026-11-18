// Click a column header to sort the paper tables.
document$.subscribe(() => {
  document.querySelectorAll("article table:not([class])").forEach((table) => new Tablesort(table));
});
