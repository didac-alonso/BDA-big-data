EN la PIPELINE d'ANALYSIS:
En els nostres ordinadors hem tingut problemes per guardar-lo amb un error amb I&O del mateix SO (Windows).
Es per això que al codi la línia de guardar està comentada, i retornem el model com una variable.

EN la PIPELINE de RUN TIME CLASSIFIER
Hi ha la línia comentada mitjançant la qual s'hauria pogut carregar el model. Tanmateix, com que no el podem
guardar, a la pràctica l'hem passat com a paràmetre a la pipeline.

EN el MAIN:

Hem generat un codi 'main.py' (adjunt a la carpeta) on es veu més clarament com solucionem aquest problema.
Adjuntem també el fitxer 'error.txt' amb l'output de l'error obtingut i el 'metrics.txt' que hauríem guardat conjuntament amb el model.
