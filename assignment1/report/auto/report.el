(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "colacl"
    "graphicx")
   (LaTeX-add-labels
    "table1"
    "table2"
    "fig"
    "table3"
    "table5")
   (LaTeX-add-bibliographies
    "article"))
 :latex)

