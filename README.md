# API using techniques from BetMaximizer project

Skal kunne:
Sende inn hjemme og bortelag og få 'Home win', 'Away win' eller 'Too close to call' tilbake.
Da må den kunne: Finne dataen for hjemmelag, finne dataen for bortelag, lage sammenligningsdata, sende sammenligningsdata til regressor, få tilbake resultatet

## TODO

- Funksjon man gir et lag så returnerer den historisk stats. Henter fra en fil
- Funksjon som lager stats over alle lag siste fem kamper. Overskriver en fil
- Trene en regressor på hver liga


### Install dependencies

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
