// Reviews write math as $...$ / $$...$$, sometimes inside raw HTML blocks that
// arithmatex never sees, so MathJax scans the whole page for both delimiter styles.
window.MathJax = {
  loader: { load: ["[tex]/cancel", "[tex]/color", "[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["cancel", "color", "boldsymbol"] },
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};

// Re-typeset after Material's instant navigation swaps the page content.
document$.subscribe(() => {
  if (!window.MathJax.startup || !window.MathJax.typesetPromise) return;
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
