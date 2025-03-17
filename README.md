# kalman

inizializzazione
- git clone https://github.com/Umaninamu/kalman
- pip install pdm (se non lo conosce)
- export PATH=$PATH:/tmp/guest/.local/bin (Se 'pip install pdm' dice di aggiungere pdm al PATH)
- pdm install (installa tutte cose)
- pdm venv activate (per attivare l'ambiente, in altre parole python3 viene chiamato usando .venv)
- pre-commit install

Prima delle modifiche
- git pull

Durante le modifiche
- pdm add «nuovo pacchetto» (ad esempio pdm add matplotlib)
- pdm remove «pacchetto vecchio» (ad esempio pdm remove matplotlib)
- pdm list (illustra tutti i pacchetti che possiedi)
- git restore «path relativa da ripristinare»

Al termine delle modifiche
- git add «path relativa da aggiungere» (mette tutte le modifiche da parte, ad esempio git add .)
- git commit -m "messaggio non vuoto" (chiude tutte le modifiche in un pacchetto) (--no-verify per eliminare il precommit)
- dopo il precommit se ci sono Failed allora ripetere con git add . ecc... (altrimenti continuare)
- git pull (si sincronizza con la repository online e confronta le versioni)
- git push (invia il pacchetto che presumibilmente è già coerente)
