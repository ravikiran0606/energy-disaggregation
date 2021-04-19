# Energy Disaggregation Backend Service

The service can be run using the following command:

```bash
$ python energy_disaggregator_service.py
```

The seervice loads a trained model and predicts the disaggregated energy for `refrigerator` and `dishawasher`. The service can be invoked using the follwing `curl` command. There is a parameter in the service called `model`. By default the service will use the `lstm` model. If the user wants to use the `cnn` model for predictions, set the value of the `model` parameter as `cnn`.

```bash
$ curl -XPOST -F file=@/Users/rijulvohra/Documents/work/USC_courses/Practicum/Project/data/redd_processed/window_3/dishwaser/test.csv "http://0.0.0.0:5600/disaggregate?model=lstm"
```

It will output a dictionary with the key being the `appliance name` and value being a list of predicted values.

**Example**

```bash
{"dishwasher": [4.4629645347595215,4.461960315704346,4.460602760314941,4.462794780731201,4.464176177978516,4.465485572814941,4.4659247398376465,4.466366291046143,4.467637538909912,4.4664506912231445,4.467343807220459,4.468395233154297,4.467753887176514,4.4684014320373535,4.467901229858398,4.466730117797852,4.46577787399292,4.466750621795654,4.46661376953125,4.465342998504639,4.465620994567871,4.466728687286377,4.465263366699219,4.46380090713501,4.463801860809326,4.465872287750244,4.467320442199707,4.469409465789795,4.471458435058594,4.470548152923584,4.46946907043457,4.473921298980713,4.474658489227295,4.470516204833984,4.467599868774414]}
```

