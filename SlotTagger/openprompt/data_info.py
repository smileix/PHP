snips_index_dict = {
    'AddToPlaylist': [1, 29, 27, 0, 25, 39],
    'BookRestaurant': [31, 3, 30, 6, 10, 26, 8, 14, 36, 12, 21, 24, 20, 13, 39],
    'GetWeather': [31, 20, 30, 16, 10, 24, 2, 38, 37, 39],
    'PlayMusic': [15, 1, 11, 35, 0, 18, 21, 7, 25, 39],
    'RateBook': [9, 17, 32, 19, 28, 34, 33, 39],
    'SearchCreativeWork': [19, 28, 39],
    'SearchScreeningEvent': [30, 23, 22, 28, 5, 24, 4, 39],
    'Std': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    }

atis_index_dict = {
    'Abbreviation': [77, 23, 73, 24, 66, 68, 36, 13, 37, 50, 3, 56, 4, 28, 83],
    'Airfare': [46, 29, 30, 47, 48, 0, 49, 1, 3, 31, 4, 5, 51, 68, 52, 53, 70, 55, 7, 57, 34, 8, 72, 10, 11, 13, 37, 14,
                18, 74, 76, 77, 60, 61, 78, 38, 80, 40, 63, 81, 26, 64, 44, 45, 28, 83],
    'Airline': [46, 30, 47, 48, 0, 49, 1, 3, 4, 5, 51, 68, 53, 71, 7, 55, 56, 34, 8, 72, 10, 13, 58, 37, 14, 16, 74, 17,
                77, 60, 80, 38, 40, 26, 65, 45, 83],
    'Flight': [46, 29, 30, 47, 48, 0, 66, 49, 1, 67, 2, 3, 31, 50, 4, 5, 6, 51, 68, 52, 53, 82, 70, 54, 69, 71, 33, 7,
               55, 56, 57, 34, 8, 72, 9, 35, 10, 11, 12, 13, 58, 14, 37, 15, 74, 16, 17, 19, 18, 75, 76, 22, 77, 59, 60,
               61, 78, 38, 80, 40, 63, 41, 25, 81, 26, 64, 44, 27, 65, 45, 28, 83],
    'GroundService': [49, 1, 50, 3, 31, 6, 54, 32, 71, 13, 74, 20, 21, 78, 79, 39, 62, 42, 40, 43, 26, 65, 45, 83],
    'Others': [46, 30, 47, 48, 0, 66, 49, 1, 50, 67, 3, 64, 31, 4, 5, 68, 52, 53, 70, 32, 71, 7, 55, 57, 56, 34, 72, 10,
               73, 11, 13, 37, 14, 74, 19, 21, 76, 77, 60, 61, 78, 38, 80, 40, 43, 26, 82, 65, 45, 28, 83]
    }

multiwoz_index_dict = {
    'BookHotel': [0, 12, 1, 11, 24, 2, 3, 5, 13, 17, 26, 25, 29, 14, 30, 7, 32, 33],
    'BookRestaurant': [11, 12, 1, 24, 3, 25, 26, 28, 6, 30, 18, 31, 21, 10, 33],
    'BookTrain': [23, 11, 1, 24, 4, 29, 15, 8, 19, 31, 33],
    'FindAttraction': [11, 1, 24, 3, 27, 26, 25, 29, 30, 18, 16, 33],
    'FindHotel': [0, 11, 1, 24, 3, 17, 5, 26, 29, 30, 7, 18, 9, 32, 33],
    'FindRestaurant': [11, 1, 12, 24, 3, 28, 29, 6, 30, 18, 21, 10, 33],
    'FindTaxi': [12, 24, 3, 25, 26, 29, 30, 18, 33],
    'FindTrain': [23, 11, 1, 12, 24, 3, 4, 25, 29, 15, 30, 8, 19, 31, 33], 'Others': [3, 22, 18, 20, 33]
    }

sgd_index_dict = {
    'Restaurants': [96, 0, 24, 26, 1, 66, 31, 32, 100, 37, 38, 41, 4, 43, 103, 104, 105, 106, 45, 76, 48, 10, 110, 12,
                    53, 56, 113, 19, 122, 123],
    'Events': [96, 21, 22, 0, 24, 25, 26, 1, 29, 99, 66, 31, 32, 100, 35, 2, 68, 37, 38, 41, 4, 70, 102, 5, 103, 63, 46,
               45, 47, 75, 76, 77, 9, 10, 95, 108, 83, 53, 54, 84, 64, 86, 56, 113, 88, 59, 114, 60, 62, 118, 91, 121,
               17, 18, 19, 122, 94, 123],
    'Music': [22, 23, 1, 66, 32, 67, 68, 102, 105, 45, 75, 76, 11, 53, 84, 59, 120, 122, 94, 123],
    'Movies': [94, 96, 22, 0, 23, 25, 26, 1, 66, 32, 100, 67, 68, 37, 38, 41, 4, 5, 102, 43, 104, 105, 45, 106, 75, 76,
               77, 48, 10, 110, 11, 83, 53, 84, 56, 113, 59, 91, 120, 18, 19, 122, 95, 123],
    'Flights': [21, 0, 24, 25, 26, 98, 99, 66, 31, 33, 34, 100, 2, 3, 101, 38, 39, 41, 4, 42, 69, 70, 6, 102, 103, 65,
                44, 7, 105, 8, 46, 77, 9, 78, 49, 95, 10, 79, 52, 81, 82, 110, 83, 55, 84, 113, 61, 114, 115, 89, 117,
                62, 14, 119, 92, 91, 16, 93, 17, 121, 18, 19, 64, 94, 123],
    'RideSharing': [94, 96, 4, 102, 66, 84, 32, 67, 56, 37, 123],
    'RentalCars': [21, 24, 25, 27, 99, 31, 35, 2, 3, 41, 4, 69, 70, 103, 73, 46, 10, 80, 109, 110, 13, 86, 60, 114, 116,
                   117, 19, 122, 123],
    'Buses': [21, 0, 24, 25, 1, 99, 31, 33, 34, 2, 38, 39, 40, 41, 4, 69, 70, 6, 103, 65, 7, 105, 8, 46, 45, 9, 78, 10,
              49, 52, 81, 82, 11, 55, 13, 113, 58, 61, 89, 62, 119, 91, 92, 120, 16, 93, 17, 18, 19, 64, 95, 123],
    'Hotels': [20, 21, 24, 25, 29, 99, 66, 31, 67, 35, 2, 3, 38, 41, 4, 69, 70, 102, 103, 105, 46, 9, 110, 84, 86, 112,
               88, 60, 115, 114, 116, 62, 118, 63, 19, 94, 123],
    'Services': [96, 30, 26, 1, 28, 29, 66, 31, 32, 100, 37, 38, 41, 4, 103, 45, 76, 10, 51, 110, 111, 12, 53, 56, 88,
                 57, 118, 63, 19, 122, 123],
    'Homes': [96, 21, 27, 36, 37, 41, 4, 70, 5, 103, 105, 73, 80, 11, 86, 56, 114, 116, 117, 120, 123],
    'Calendar': [96, 0, 26, 32, 37, 38, 41, 4, 103, 73, 47, 10, 108, 80, 54, 85, 56, 57, 18, 95, 123],
    'Weather': [96, 0, 24, 25, 1, 29, 66, 32, 100, 67, 2, 101, 68, 37, 38, 41, 4, 69, 102, 103, 44, 7, 46, 45, 75, 79,
                10, 95, 80, 84, 56, 113, 88, 59, 62, 14, 118, 91, 93, 63, 18, 19, 94, 123],
    'Travel': [94, 96, 22, 0, 23, 24, 25, 29, 66, 31, 100, 67, 2, 68, 37, 41, 4, 102, 103, 105, 46, 75, 9, 10, 110, 84,
               113, 88, 59, 62, 118, 91, 63, 18, 19, 122, 95, 123],
    'Others': [96, 22, 97, 23, 0, 24, 25, 66, 100, 101, 38, 41, 4, 42, 5, 71, 72, 103, 44, 7, 105, 74, 75, 107, 50, 10,
               79, 110, 11, 53, 64, 87, 56, 59, 90, 14, 15, 91, 120, 93, 18, 122, 95, 123]
    }

dataset2index = {
    "SNIPS": snips_index_dict,
    "ATIS": atis_index_dict,
    "MultiWOZ": multiwoz_index_dict,
    "SGD": sgd_index_dict,
    }

snips_valid_label = [
    'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 'I-geographic_poi',
    'B-facility', 'I-facility', 'B-movie_name', 'I-movie_name', 'B-location_name', 'I-location_name',
    'B-restaurant_name', 'I-restaurant_name', 'B-track', 'I-track', 'B-restaurant_type', 'I-restaurant_type',
    'B-object_part_of_series_type', 'I-object_part_of_series_type', 'B-country', 'I-country', 'B-service', 'I-service',
    'B-poi', 'I-poi', 'B-party_size_description', 'I-party_size_description', 'B-served_dish', 'I-served_dish',
    'B-genre', 'I-genre', 'B-current_location', 'I-current_location', 'B-object_select', 'I-object_select', 'B-album',
    'I-album', 'B-object_name', 'I-object_name', 'B-state', 'I-state', 'B-sort', 'I-sort', 'B-object_location_type',
    'I-object_location_type', 'B-movie_type', 'I-movie_type', 'B-spatial_relation', 'I-spatial_relation', 'B-artist',
    'I-artist', 'B-cuisine', 'I-cuisine', 'B-entity_name', 'I-entity_name', 'B-object_type', 'I-object_type',
    'B-playlist_owner', 'I-playlist_owner', 'B-timeRange', 'I-timeRange', 'B-city', 'I-city', 'B-rating_value',
    'B-best_rating', 'B-rating_unit', 'B-year', 'B-party_size_number', 'B-condition_description',
    'B-condition_temperature', 'O']

atis_valid_labels = [
    'I-flight_time', 'B-restriction_code', 'I-fare_basis_code', 'I-transport_type',
    'B-return_time.period_mod', 'B-toloc.country_name', 'B-flight', 'I-toloc.state_name', 'B-fromloc.state_code',
    'B-arrive_time.start_time', 'I-airline_name', 'B-arrive_time.period_mod', 'I-return_date.day_number',
    'I-meal_description', 'B-depart_time.time_relative', 'B-cost_relative', 'B-depart_date.month_name',
    'B-aircraft_code', 'B-toloc.city_name', 'B-depart_date.year', 'B-connect', 'I-return_date.date_relative',
    'B-airport_name', 'B-meal_description', 'B-depart_date.day_name', 'B-flight_stop', 'I-airport_name', 'B-class_type',
    'B-arrive_time.end_time', 'B-days_code', 'I-round_trip', 'B-state_code', 'B-return_time.period_of_day',
    'B-flight_mod', 'B-depart_time.period_mod', 'B-arrive_time.time', 'I-stoploc.city_name',
    'B-return_date.date_relative', 'B-period_of_day', 'B-return_date.month_name', 'B-toloc.state_name', 'B-day_name',
    'B-stoploc.city_name', 'I-today_relative', 'B-toloc.state_code', 'B-meal_code', 'B-stoploc.airport_code',
    'B-arrive_time.period_of_day', 'B-arrive_date.day_number', 'B-toloc.airport_name', 'B-stoploc.state_code',
    'I-city_name', 'B-state_name', 'B-depart_date.day_number', 'B-return_date.today_relative',
    'I-return_date.today_relative', 'I-flight_number', 'I-fromloc.airport_name', 'I-depart_time.time',
    'I-depart_time.period_of_day', 'B-today_relative', 'B-arrive_date.day_name', 'I-arrive_time.end_time',
    'I-depart_time.time_relative', 'B-fromloc.airport_code', 'I-toloc.city_name', 'B-depart_time.end_time',
    'B-flight_number', 'I-fromloc.city_name', 'B-transport_type', 'B-round_trip', 'I-flight_stop',
    'B-depart_time.period_of_day', 'I-arrive_time.period_of_day', 'B-compartment', 'B-flight_time',
    'B-return_date.day_number', 'B-month_name', 'B-airport_code', 'B-depart_time.start_time', 'I-restriction_code',
    'B-mod', 'I-arrive_time.start_time', 'B-time_relative', 'I-depart_date.day_number', 'B-economy', 'I-state_name',
    'B-arrive_date.date_relative', 'B-fromloc.city_name', 'B-depart_date.today_relative', 'B-fare_amount',
    'I-fromloc.state_name', 'B-arrive_time.time_relative', 'I-arrive_time.time', 'I-toloc.airport_name', 'I-economy',
    'I-arrive_time.time_relative', 'I-time', 'I-depart_date.today_relative', 'B-depart_date.date_relative',
    'I-fare_amount', 'B-stoploc.airport_name', 'B-booking_class', 'I-meal_code', 'B-fromloc.state_name', 'B-meal',
    'B-arrive_date.month_name', 'I-depart_time.start_time', 'B-airline_code', 'B-arrive_date.today_relative',
    'I-flight_mod', 'B-or', 'B-toloc.airport_code', 'I-cost_relative', 'I-class_type', 'B-time',
    'I-arrive_date.day_number', 'B-fromloc.airport_name', 'B-day_number', 'B-airline_name', 'I-depart_time.end_time',
    'B-return_date.day_name', 'B-city_name', 'B-depart_time.time', 'B-fare_basis_code', 'B-flight_days', 'O']

multiwoz_valid_labels = [
    'B-train-arriveby', 'B-attraction-name', 'I-restaurant-name', 'B-train-leaveat',
    'B-taxi-departure', 'B-hotel-bookpeople', 'B-hotel-type', 'B-restaurant-pricerange', 'B-train-departure',
    'B-train-bookpeople', 'I-train-arriveby', 'B-attraction-area', 'I-attraction-name', 'B-bus-departure',
    'B-restaurant-bookday', 'B-restaurant-food', 'B-hotel-stars', 'B-taxi-arriveby', 'B-train-destination',
    'I-train-leaveat', 'B-restaurant-name', 'B-hotel-bookstay', 'B-train-day', 'I-train-departure',
    'B-hospital-department', 'B-restaurant-booktime', 'B-hotel-pricerange', 'I-hotel-name', 'I-hotel-type',
    'B-bus-destination', 'I-taxi-leaveat', 'B-hotel-area', 'B-attraction-type', 'I-train-destination',
    'I-hospital-department', 'B-restaurant-area', 'I-restaurant-food', 'I-attraction-type', 'I-hotel-stars',
    'B-hotel-internet', 'I-restaurant-booktime', 'I-taxi-departure', 'B-taxi-leaveat', 'B-restaurant-bookpeople',
    'I-taxi-arriveby', 'B-hotel-name', 'B-taxi-destination', 'B-hotel-bookday', 'I-taxi-destination', 'B-hotel-parking',
    'O']

sgd_valid_labels = [
    'I-venue_address', 'B-start_date', 'I-destination_city', 'B-return_date', 'B-price_per_day',
    'I-destination_station_name', 'B-total', 'B-venue_address', 'I-inbound_arrival_time', 'I-check_in_date',
    'I-outbound_arrival_time', 'B-leaving_time', 'B-departure_time', 'I-venue', 'B-journey_start_time',
    'I-date_of_journey', 'I-alarm_name', 'B-track', 'B-available_end_time', 'B-alarm_time', 'B-event_date',
    'B-pickup_date', 'B-destination_city', 'B-place_name', 'B-dropoff_date', 'B-from_city', 'I-new_alarm_name',
    'I-origin_airport', 'I-event_location', 'I-inbound_departure_time', 'B-check_out_date', 'B-address',
    'I-leaving_date', 'I-from_location', 'B-pickup_time', 'B-movie_name', 'B-rent', 'B-event_location', 'B-rating',
    'B-destination_station_name', 'B-fare', 'I-doctor_name', 'I-category', 'I-destination_airport', 'B-ride_fare',
    'I-event_time', 'I-show_time', 'B-show_date', 'B-contact_name', 'I-artist', 'I-date', 'B-percent_rating',
    'B-destination_airport_name', 'I-street_address', 'B-outbound_arrival_time', 'B-destination_airport',
    'I-subcategory', 'B-category', 'B-balance', 'B-movie_title', 'B-receiver', 'B-outbound_departure_time',
    'I-property_name', 'B-leaving_date', 'I-appointment_date', 'B-transfer_amount', 'B-average_rating', 'B-time',
    'B-director', 'I-available_end_time', 'B-genre', 'B-directed_by', 'I-alarm_time', 'I-actors', 'B-to_station',
    'I-amount', 'I-phone_number', 'I-pickup_time', 'I-origin_city', 'B-pickup_city', 'B-transfer_time',
    'I-transfer_amount', 'I-therapist_name', 'B-location', 'B-subcategory', 'I-restaurant_name', 'I-theater_name',
    'I-to', 'B-event_name', 'I-location', 'B-from', 'B-amount', 'I-available_start_time', 'I-check_out_date', 'I-from',
    'B-stay_length', 'I-hotel_name', 'I-starring', 'B-date_of_journey', 'B-venue', 'I-genre', 'B-departure_date',
    'I-to_city', 'B-origin', 'B-restaurant_name', 'B-stylist_name', 'B-pickup_location', 'I-event_name',
    'I-origin_airport_name', 'B-address_of_location', 'I-event_date', 'B-approximate_ride_duration', 'B-hotel_name',
    'B-song_name', 'I-director', 'I-area', 'B-wind', 'I-show_date', 'I-movie_name', 'I-to_station', 'I-start_date',
    'B-new_alarm_time', 'I-outbound_departure_time', 'I-address', 'I-from_station', 'I-new_alarm_time',
    'B-price_per_night', 'I-return_date', 'B-appointment_time', 'B-check_in_date', 'I-dentist_name',
    'I-address_of_location', 'B-recipient_account_name', 'I-pickup_date', 'B-cast', 'B-number_of_days',
    'B-account_balance', 'I-city', 'I-album', 'I-to_location', 'B-new_alarm_name', 'B-destination',
    'B-origin_station_name', 'B-actors', 'B-aggregate_rating', 'B-alarm_name', 'B-temperature', 'B-street_address',
    'I-car_name', 'I-city_of_event', 'I-cast', 'B-album', 'I-destination_airport_name', 'B-price',
    'I-journey_start_time', 'B-end_date', 'B-theater_name', 'B-attraction_name', 'I-attraction_name', 'I-from_city',
    'I-place_name', 'B-appointment_date', 'B-inbound_arrival_time', 'I-movie_title', 'B-show_time', 'B-wait_time',
    'I-song_name', 'I-departure_time', 'I-visit_date', 'B-total_price', 'B-title', 'B-starring', 'I-pickup_city',
    'B-visit_date', 'I-track', 'B-price_per_ticket', 'B-event_time', 'I-destination', 'I-departure_date',
    'I-leaving_time', 'B-recipient_name', 'I-cuisine', 'B-to_location', 'B-cuisine', 'B-phone_number', 'I-end_date',
    'B-available_start_time', 'B-where_to', 'B-origin_airport', 'B-date', 'B-from_location', 'B-humidity',
    'I-pickup_location', 'I-origin', 'B-area', 'I-stylist_name', 'B-property_name', 'B-origin_city', 'B-dentist_name',
    'B-precipitation', 'I-title', 'B-city_of_event', 'I-recipient_name', 'B-inbound_departure_time',
    'B-origin_airport_name', 'B-doctor_name', 'B-car_name', 'I-dropoff_date', 'I-time', 'I-directed_by', 'B-to',
    'B-to_city', 'I-origin_station_name', 'I-appointment_time', 'B-artist', 'B-from_station', 'B-city', 'I-where_to',
    'B-therapist_name', 'O']

dataset2valid_labels = {
    "SNIPS": snips_valid_label,
    "ATIS": atis_valid_labels,
    "MultiWOZ": multiwoz_valid_labels,
    "SGD": sgd_valid_labels,
    }


# 此处的domain2desp是数据处理时所需要用到的，domain name应该与数据集中的标注相同。
snips_domain2desp = {
    "AddToPlaylist": "add to playlist", "BookRestaurant": "reserve restaurant", "GetWeather": "get weather",
    "PlayMusic": "play music", "RateBook": "rate book", "SearchCreativeWork": "search creative work",
    "SearchScreeningEvent": "search screening event"
    }

# 此处的domain2desp是数据处理时所需要用到的，domain name应该与数据集中的标注相同。
# todo others的处理，是用原先的domain name，还是用others代替，亦或者不用domain描述，只对snips用。先做实验看看效果。
atis_domain2desp = {
    'atis_abbreviation': "abbreviation",
    'atis_airfare': 'airfare',
    'atis_airline': 'airline',
    'atis_flight': 'flight',
    'atis_ground_service': 'ground service',
    # 如果不在字典中，则获取默认值 others
    'Others': 'others'
    }

# 此处的domain2desp是数据处理时所需要用到的，domain name应该与数据集中的标注相同。
multiwoz_domain2desp = {
    'book_hotel': 'reserve hotel',
    'book_restaurant': 'reserve restaurant',
    'book_train': 'reserve train',
    'find_attraction': 'find attraction',
    'find_hotel': 'find hotel',
    'find_restaurant': 'find restaurant',
    'find_taxi': 'find taxi',
    'find_train': 'find train',
    # 如果不在字典中，则获取默认值 others
    'Others': 'other',
    }

# 此处的domain2desp是数据处理时所需要用到的，domain name应该与数据集中的标注相同。
sgd_domain2desp = {
    'Restaurants': 'restaurants',
    'Events': 'events',
    'Music': 'music',
    'Movies': 'movies',
    'Flights': 'flights',
    'RideSharing': 'ride sharing',
    'RentalCars': 'rental cars',
    'Buses': 'buses',
    'Services': 'services',
    'Homes': 'homes',
    'Calendar': 'calendar',
    'Weather': 'weather',
    'Travel': 'travel',
    # 如果不在字典中，则获取默认值 others
    'Others': 'others',
    }

dataset2domain_desp = {
    "SNIPS": snips_domain2desp,
    "ATIS": atis_domain2desp,
    "MultiWOZ": multiwoz_domain2desp,
    "SGD": sgd_domain2desp,
    }

snips_slot_eg = {
    # AddToPlaylist
    "music_item": ["song", "track"],
    "playlist_owner": ["my", "donna s"],
    "entity_name": ["the crabfish", "natasha"],
    "playlist": ["quiero playlist", "workday lounge"],
    "artist": ["lady bunny", "lisa dalbello"],
    # BookRestaurant
    "city": ["north lima", "falmouth"],
    "facility": ["smoking room", "indoor"],
    "timeRange": ["9 am", "january the twentieth"],
    "restaurant_name": ["the maisonette", "robinson house"],
    "country": ["dominican republic", "togo"],
    "cuisine": ["ouzeri", "jewish"],
    "restaurant_type": ["tea house", "tavern"],
    "served_dish": ["wings", "cheese fries"],
    "party_size_number": ["seven", "one"],
    "poi": ["east brady", "fairview"],
    "sort": ["top-rated", "highly rated"],
    "spatial_relation": ["close", "faraway"],
    "state": ["sc", "ut"],
    "party_size_description": ["me and angeline", "my colleague and i"],
    # GetWeather
    "current_location": ["current spot", "here"],
    "geographic_poi": ["bashkirsky nature reserve", "narew national park"],
    "condition_temperature": ["chillier", "hot"],
    "condition_description": ["humidity", "depression"],
    # PlayMusic
    "genre": ["techno", "pop"],
    "service": ["spotify", "groove shark"],
    "year": ["2005", "1993"],
    "album": ["allergic", "secrets on parade"],
    "track": ["in your eyes", "the wizard and i"],
    # RateBook
    "object_part_of_series_type": ["series", "saga"],
    "object_select": ["this", "current"],
    "rating_value": ["1", "four"],
    "object_name": ["american tabloid", "my beloved world"],
    "object_type": ["book", "novel"],
    "rating_unit": ["points", "stars"],
    "best_rating": ["6", "5"],
    # SearchCreativeWork
    # SearchScreeningEvent
    "movie_type": ["animated movies", "films"],
    "object_location_type": ["movie theatre", "cinema"],
    "location_name": ["amc theaters", "wanda group"],
    "movie_name": ["on the beat", "for lovers only"]
}

unseen_slot_based_dataset_domain = {
    'ATIS': {
        'Abbreviation': ['meal_code', 'booking_class', 'days_code'], 'Airfare': [], 'Airline': [],
        'Flight': ['arrive_time.end_time', 'return_date.today_relative', 'stoploc.state_code', 'toloc.country_name',
                   'stoploc.airport_name', 'arrive_date.today_relative', 'return_time.period_of_day', 'flight',
                   'compartment', 'return_time.period_mod', 'arrive_time.start_time', 'stoploc.airport_code',
                   'arrive_time.period_mod'],
        'GroundService': ['time', 'day_number', 'time_relative', 'month_name', 'today_relative'], 'Others': [],
    }, 'SNIPS': {
        'AddToPlaylist': ['playlist_owner', 'entity_name'],
        'BookRestaurant': ['served_dish', 'party_size_number', 'facility', 'party_size_description', 'cuisine',
                             'restaurant_type', 'restaurant_name', 'poi'],
        'GetWeather': ['condition_description', 'condition_temperature', 'current_location', 'geographic_poi'],
        'PlayMusic': ['track', 'album', 'year', 'genre', 'service'],
        'RateBook': ['best_rating', 'object_part_of_series_type', 'rating_unit', 'object_select', 'rating_value'],
        'SearchCreativeWork': [],
        'SearchScreeningEvent': ['object_location_type', 'movie_type', 'movie_name', 'location_name'],
    },

    'MultiWOZ': {
        'BookHotel': ['hotel-bookpeople', 'hotel-bookday', 'hotel-bookstay'], 'BookRestaurant': [],
        'BookTrain': [], 'FindAttraction': ['attraction-type', 'attraction-area'], 'FindHotel': [],
        'FindRestaurant': ['hotel-internet'], 'FindTaxi': [], 'FindTrain': [],
        'Others': ['bus-departure', 'bus-destination'],
    },
}
