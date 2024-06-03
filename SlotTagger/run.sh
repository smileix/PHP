# std
python main.py --config_yaml experiments/snips.yaml --dataset.domain Std --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/atis.yaml --dataset.domain Std --environment.cuda_visible_devices 3
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain Std --environment.cuda_visible_devices 0

# train snips
# 0-shot
#<editor-fold desc="snips 0-shot">
python main.py --config_yaml experiments/snips.yaml --dataset.domain AddToPlaylist_0 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain BookRestaurant_0 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain GetWeather_0 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain PlayMusic_0 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain RateBook_0 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchCreativeWork_0 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchScreeningEvent_0 --environment.cuda_visible_devices 0
#</editor-fold>

#<editor-fold desc="snips few-shot">
##### 20-shot
python main.py --config_yaml experiments/snips.yaml --dataset.domain AddToPlaylist_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain BookRestaurant_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain GetWeather_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain PlayMusic_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain RateBook_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchCreativeWork_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchScreeningEvent_20 --environment.cuda_visible_devices 0

####### 50-shot
python main.py --config_yaml experiments/snips.yaml --dataset.domain AddToPlaylist_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain BookRestaurant_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain GetWeather_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain PlayMusic_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain RateBook_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchCreativeWork_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchScreeningEvent_50 --environment.cuda_visible_devices 0


# 100-shot
python main.py --config_yaml experiments/snips.yaml --dataset.domain AddToPlaylist_100 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain BookRestaurant_100 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain GetWeather_100 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain PlayMusic_100 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain RateBook_100 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchCreativeWork_100 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchScreeningEvent_100 --environment.cuda_visible_devices 0

# 200-shot
python main.py --config_yaml experiments/snips.yaml --dataset.domain AddToPlaylist_200 --environment.cuda_visible_devices 1
python main.py --config_yaml experiments/snips.yaml --dataset.domain BookRestaurant_200 --environment.cuda_visible_devices 1
python main.py --config_yaml experiments/snips.yaml --dataset.domain GetWeather_200 --environment.cuda_visible_devices 1
python main.py --config_yaml experiments/snips.yaml --dataset.domain PlayMusic_200 --environment.cuda_visible_devices 1
python main.py --config_yaml experiments/snips.yaml --dataset.domain RateBook_200 --environment.cuda_visible_devices 1
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchCreativeWork_200 --environment.cuda_visible_devices 1
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchScreeningEvent_200 --environment.cuda_visible_devices 1

# 400-shot
python main.py --config_yaml experiments/snips.yaml --dataset.domain AddToPlaylist_400 --environment.cuda_visible_devices 2
python main.py --config_yaml experiments/snips.yaml --dataset.domain BookRestaurant_400 --environment.cuda_visible_devices 2
python main.py --config_yaml experiments/snips.yaml --dataset.domain GetWeather_400 --environment.cuda_visible_devices 2
python main.py --config_yaml experiments/snips.yaml --dataset.domain PlayMusic_400 --environment.cuda_visible_devices 2
python main.py --config_yaml experiments/snips.yaml --dataset.domain RateBook_400 --environment.cuda_visible_devices 2
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchCreativeWork_400 --environment.cuda_visible_devices 2
python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchScreeningEvent_400 --environment.cuda_visible_devices 2

## test
## 0s
#python main.py --config_yaml experiments/snips.yaml --dataset.domain AddToPlaylist_0 --test AddToPlaylist_0
#python main.py --config_yaml experiments/snips.yaml --dataset.domain BookRestaurant_0 --test BookRestaurant_0
#python main.py --config_yaml experiments/snips.yaml --dataset.domain GetWeather_0 --test GetWeather_0
#python main.py --config_yaml experiments/snips.yaml --dataset.domain PlayMusic_0 --test PlayMusic_0
#python main.py --config_yaml experiments/snips.yaml --dataset.domain RateBook_0 --test RateBook_0
#python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchCreativeWork_0 --test SearchCreativeWork_0
#python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchScreeningEvent_0 --test SearchScreeningEvent_0
#
## 20s
#python main.py --config_yaml experiments/snips.yaml --dataset.domain AddToPlaylist_20 --test AddToPlaylist_20
#python main.py --config_yaml experiments/snips.yaml --dataset.domain BookRestaurant_20 --test BookRestaurant_20
#python main.py --config_yaml experiments/snips.yaml --dataset.domain GetWeather_20 --test GetWeather_20
#python main.py --config_yaml experiments/snips.yaml --dataset.domain PlayMusic_20 --test PlayMusic_20
#python main.py --config_yaml experiments/snips.yaml --dataset.domain RateBook_20 --test RateBook_20
#python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchCreativeWork_20 --test SearchCreativeWork_20
#python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchScreeningEvent_20 --test SearchScreeningEvent_20
#
## 50s
#python main.py --config_yaml experiments/snips.yaml --dataset.domain AddToPlaylist_50 --test AddToPlaylist_50
#python main.py --config_yaml experiments/snips.yaml --dataset.domain BookRestaurant_50 --test BookRestaurant_50
#python main.py --config_yaml experiments/snips.yaml --dataset.domain GetWeather_50 --test GetWeather_50
#python main.py --config_yaml experiments/snips.yaml --dataset.domain PlayMusic_50 --test PlayMusic_50
#python main.py --config_yaml experiments/snips.yaml --dataset.domain RateBook_50 --test RateBook_50
#python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchCreativeWork_50 --test SearchCreativeWork_50
#python main.py --config_yaml experiments/snips.yaml --dataset.domain SearchScreeningEvent_50 --test SearchScreeningEvent_50

#</editor-fold>

# train atis
#<editor-fold desc="atis">
python main.py --config_yaml experiments/atis.yaml --dataset.domain Abbreviation --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/atis.yaml --dataset.domain Airfare --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/atis.yaml --dataset.domain Airline --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/atis.yaml --dataset.domain Flight --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/atis.yaml --dataset.domain GroundService --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/atis.yaml --dataset.domain Others --environment.cuda_visible_devices 0
#</editor-fold>

#<editor-fold desc="multiwoz">
# train multiwoz 0s
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain BookHotel_0 --environment.cuda_visible_devices 2
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain BookRestaurant_0 --environment.cuda_visible_devices 2
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain BookTrain_0 --environment.cuda_visible_devices 2
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindAttraction_0 --environment.cuda_visible_devices 3
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindHotel_0 --environment.cuda_visible_devices 3
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindRestaurant_0 --environment.cuda_visible_devices 3
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindTaxi_0 --environment.cuda_visible_devices 3
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindTrain_0 --environment.cuda_visible_devices 3
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain Others_0 --environment.cuda_visible_devices 3

# 20s
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain BookHotel_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain BookRestaurant_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain BookTrain_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindAttraction_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindHotel_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindRestaurant_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindTaxi_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindTrain_20 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain Others_20 --environment.cuda_visible_devices 0

# 50s
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain BookHotel_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain BookRestaurant_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain BookTrain_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindAttraction_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindHotel_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindRestaurant_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindTaxi_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain FindTrain_50 --environment.cuda_visible_devices 0
python main.py --config_yaml experiments/multiwoz.yaml --dataset.domain Others_50 --environment.cuda_visible_devices 0
