### Readme is under construction.

* infobert and ascc is derived from textdefender(https://github.com/RockyLzy/TextDefender)
* vibert is derived from vibert(https://github.com/rabeehk/vibert)
* disentangle is derived from dib(https://github.com/PanZiqiAI/disentangled-information-bottleneck)
* textfooler and deepwordbug is using implementation of textattack(https://github.com/QData/TextAttack)

A big thanks to this project mentioned above!

We don't spend much time to reconstruct the style/strucure of this project. It may looks like a mess. But it's pain free to add your own model in our workflow. You only need modify the load model method to load your own model. And make sure your model have a forward function which return the logits.
