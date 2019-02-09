# Urban Rural Binary Image Classifier

##### Verbose Tile
Binary Image Classification of Urban and Rural Aerial Imagery Using Convolutional Neural Networks
##### Input Parameters
Aerial imagery as a .jpg file save

&nbsp;&nbsp;&nbsp;&nbsp;See my other repository, Aerial Image Retrieval
##### Output
In console, either 'urban' or 'rural'
##### Usage
In the command line,

&nbsp;&nbsp;&nbsp;&nbsp;To generate a model after specifying parameters:
```sh
$ python -c 'import urban_rural_classification.py; print urban_rural_classification.generate_classifier()'
```

&nbsp;&nbsp;&nbsp;&nbsp;To return an output for an image:
```sh
$ python -c 'import urban_rural_classification.py; print urban_rural_classification.prediction()'
```

License
----
MIT
