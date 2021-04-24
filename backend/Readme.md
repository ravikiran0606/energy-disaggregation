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

## Service for Forecasting Energy

```bash
$ curl -XPOST -F file=@/Users/rijulvohra/Documents/work/USC_courses/Practicum/Project/data/forecasting_sample_data/dishwaser/ \ h1_train.csv "http://0.0.0.0:5600/forecast?time=24"
```

The parameter `time` in the API is an integer and denotes for how many hours do you want to forecast the energy consumption.

**Example**

The service returns a dictionary with key being a `string` of the `datetime` and the value being a `dictionary`. The value has two keys:

* `flag`: Representing whether the corresponding `output` value is historical data or forecasted data. If `flag:0` means corresponding `output` is historical data and if `flag:1` means corresponding `output is forecasted data. 

```bash
{"2011-05-13 12:00:00":{"flag":0,"output":0.0},"2011-05-13 13:00:00": {"flag":0,"output":0.0291666666666666},"2011-05-13 14:00:00":{"flag":0,"output":0.0166666666666666},"2011-05-13 15:00:00":{"flag":0,"output":0.0},"2011-05-13 16:00:00":{"flag":0,"output":0.0166666666666666},"2011-05-13 17:00:00":{"flag":0,"output":0.025},"2011-05-13 18:00:00":{"flag":0,"output":0.0},"2011-05-13 19:00:00":{"flag":0,"output":0.05},"2011-05-13 20:00:00":{"flag":0,"output":0.0111166666666666},"2011-05-13 21:00:00":{"flag":0,"output":0.0},"2011-05-13 22:00:00":{"flag":1,"output":0.017333760269769224},"2011-05-13 23:00:00":{"flag":1,"output":0.016320518314321637},"2011-05-14 00:00:00":{"flag":1,"output":0.015895023459064943},"2011-05-14 01:00:00":{"flag":1,"output":0.016228474323189114},"2011-05-14 02:00:00":{"flag":1,"output":0.015369398667316783},"2011-05-14 03:00:00":{"flag":1,"output":0.015316523588418958},"2011-05-14 04:00:00":{"flag":1,"output":0.01710844554512047},"2011-05-14 05:00:00":{"flag":1,"output":0.015638334336305987},"2011-05-14 06:00:00":{"flag":1,"output":0.016975817428253603},"2011-05-14 07:00:00":{"flag":1,"output":0.016523365397827353},"2011-05-14 08:00:00":{"flag":1,"output":0.014523389815416593},"2011-05-14 09:00:00":{"flag":1,"output":0.015108985655213524},"2011-05-14 10:00:00":{"flag":1,"output":0.01587311919936872},"2011-05-14 11:00:00":{"flag":1,"output":0.015904510504604723},"2011-05-14 12:00:00":{"flag":1,"output":0.015878743397550802},"2011-05-14 13:00:00":{"flag":1,"output":0.015870112773337133},"2011-05-14 14:00:00":{"flag":1,"output":0.0158203125191589},"2011-05-14 15:00:00":{"flag":1,"output":0.0159194641492793},"2011-05-14 16:00:00":{"flag":1,"output":0.01588420432634407},"2011-05-14 17:00:00":{"flag":1,"output":0.01585022626951686},"2011-05-14 18:00:00":{"flag":1,"output":0.015846485042181953},"2011-05-14 19:00:00":{"flag":1,"output":0.015727333627462717},"2011-05-14 20:00:00":{"flag":1,"output":0.01570331131258887},"2011-05-14 21:00:00":{"flag":1,"output":0.015787091991692384}}
```

