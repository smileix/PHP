"""
This file contains the logic for loading data for all typing tasks.
"""
import os
import random
from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_info import dataset2domain_desp


class SNIPSProcessor(DataProcessor):
    """
    Snips数据集的处理

    Examples:
        from openprompt.data_utils.typing_dataset import PROCESSORS

        base_path = "datasets/Typing"
        dataset_name = "Snips"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 数据还需确认
        assert processor.get_labels() == [] label也需要确认
        assert dev_dataset[0].text_a == demo
        assert dev_dataset[0].meta["entity"] == "Skyfox"
        assert dev_dataset[0].label == 30
        验证一下数据处理的是否正确

        TODO 这种处理方式，以及之后采用的模板，大致是<X>, in this sentence, <entity> is <mask>,问槽是什么类型的实体
             比如，book the hat for my classmates，in this sentence, the hat is restaurant name. 可能更契合Cloze propmting
             PS：可以效仿RCSF，在template里面添加描述或示例，大致是<X>, in this sentence, <entity> is <mask> like <example1> or <example2>
             此处的mask应为label of entity
             以该方式处理训练集没问题，但是对于验证集与测试集的处理，则有些问题。因为该处理方式需要提前知道句子的标注信息，知道实体的具体位置。
             而验证集与测试集的标注信息，对模型来说是未知的，至少测试集的标注信息对于模型来说是未知的。
             以该种方法处理验证集与测试集，需要引入另一个标注任务，预测每个token的BIO标签，之后再预测具体的实体类型。或者实现类似功能。也是由粗到细。
             这种处理方式更兼容现有的框架。目前先不考虑BIO标注的事情。
             或者，（只）在inference的时候遍历所有span。也就是说，训练集的方式不变，但是验证集与测试集遍历所有span。

        TODO 或者就学习RCSF，改变template的形式，反着问。采用的模板大概是，<X>, in this sentence, what is <label of entity>? <mask>
             此处的mask应为entity。更改mask，也需要更改对应的verbalizer。用这种格式，确实更契合抽取式QA任务。
             book the hat for my classmates，in this sentence, what is the restaurant name? the hat。可能更契合Prefix prompting。
             使用这种方法的话，可能会有空问题，也就是答案为None。

    """

    def __init__(self):
        super().__init__()
        self.labels = ['playlist', 'music_item', 'geographic_poi', 'facility', 'movie_name', 'location_name',
                       'restaurant_name', 'track', 'restaurant_type', 'object_part_of_series_type', 'country',
                       'service', 'poi', 'party_size_description', 'served_dish', 'genre', 'current_location',
                       'object_select', 'album', 'object_name', 'state', 'sort', 'object_location_type', 'movie_type',
                       'spatial_relation', 'artist', 'cuisine', 'entity_name', 'object_type', 'playlist_owner',
                       'timeRange', 'city', 'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number',
                       'condition_description', 'condition_temperature', 'O']

    def get_examples(self, data_dir, split=None, domain=None, use_true_bio_label=False):
        # 这里的domain为str，形式为AddToPlaylist_0等
        assert domain is not None
        path = os.path.join(data_dir, "{}/{}.tsv".format(domain, split))
        # path = data_dir
        with open(path, encoding='utf8') as f:
            data = SNIPSProcessor.load_data(f, split, use_true_bio_label)
            examples = []
            cnt = 0
            for idx, (xs, ys, spans, domain) in enumerate(data):
                # todo 可以尝试需要加些负样本，以缓解错误传递的问题
                # 可以提前对spans处理，增加负样本的spans
                # if split == 'train':
                #     if cnt % 8 == 0:
                #         entity_index = []
                #         for span in spans:
                #             entity_index.extend(span)
                #         entity_index_set = set(entity_index)
                #         all_index_set = set(list(range(len(xs))))
                #         non_entity_index_list = list(all_index_set - entity_index_set)
                #         [random_non_entity_index] = random.choices(non_entity_index_list)
                #         span_neg = [random_non_entity_index, random_non_entity_index]
                #         spans.append(span_neg)
                #     cnt += 1

                for span in spans:
                    text_a = " ".join(xs)
                    # 判断是否为train
                    if split == 'train':
                        # slot = ys[span[0]].split('-')[1]
                        meta = {
                            "entity": " ".join(xs[span[0]: span[1] + 1]),
                            "domain": dataset2domain_desp['SNIPS'][domain],
                            }  # "eg": " or ".join(slot_eg[ys[span[0]].split('-')[1]]),
                    else:

                        meta = {
                            "entity": " ".join(xs[span[0]: span[1] + 1]),
                            "domain": dataset2domain_desp['SNIPS'][domain],
                            }

                    # 对于验证集与测试集，如果span对应的label为O时，则对应的span label也为O，如果不为O，那么去掉B-或I-前缀赋给span label
                    if ys[span[0]].startswith('O'):
                        span_label = self.get_label_id('O')
                    else:
                        span_label = self.get_label_id(ys[span[0]][2:])

                    #  将整个句子的原始标签ys以及跨度信息spans传进batch中， 简单点可以一起放进label里面传递
                    label = {"ys": ys, "span": span, "span_label": span_label}
                    # ys 是 list[int], span 也是list[int]， label是int
                    example = InputExample(guid=str(idx), text_a=text_a, meta=meta, label=label)
                    examples.append(example)

            return examples

    @staticmethod
    def load_data(file, data_split, use_true_bio_label=False):
        # data的list，长度为数据集的条数，每个元素是tuple，
        # tuple的长度为3，每个元素是list，分别是输入x、输出y与每个实体的跨度，
        # 第三个是spans，表示每个entity的跨度，用一个长为2的list来表示，形如[[2, 2], [6, 7], [9, 10]],表示有3个实体，及其对应的跨度
        data = []
        for line in file.readlines():
            splits = line.strip().split('\t')
            raw_xs = splits[0]
            raw_ys = splits[1]
            domain = splits[2]

            xs = raw_xs.split()
            ys = raw_ys.split()

            spans = []
            start = 0
            end = 1

            # TODO 后续添加dev的判断，需要在此之前给dev也打上伪标签
            if data_split == 'dev' or data_split == 'test':
                # if data_split == 'test':
                if not use_true_bio_label:
                    assert len(splits) == 4
                    raw_pseudos = splits[3]
                    pseudos = raw_pseudos.split()
                    ys = pseudos

            # 真实标签都是合法的，不会存在I开头的情况。
            # 伪标签则会存在不合法的情况
            # 对于不合法的情况，主要是指I开头，目前的处理方式是变成B，但是会带来其他影响
            ## 如果真实标签是O，伪标签是I，改成B之后就会出现已经发生的情况，因为标签空间里面没有O，在不改标签空间的前提下，就需要随便分配一个标签，必定会影响最后的指标
            ## 如果真实标签是B，伪标签是I，改成B之后就相当于是恢复正常了。
            # 如果处理方式更改，变成将不合法的I开头，改成O，则
            ## 如果真实标签是O，则无影响
            ## 如果真实标签是B，则会导致少预测一个实体。如果是BIII的情况预测成了IIII，则会导致整个实体预测出错，因为会处理成OOOO,
            # 因此，改进处理方式，对于单个的I，如OIO，OIB，（可能也要考虑最末尾等，我们将其合法化，处理成O，其他情况，也就是有两个及以上连续的I，如OII，我们将其第一个I处理成B

            # end< len ys，end的初始值是1，没有考虑到len = 1的情况。
            if len(ys) == 1 and ys[0].startswith('B'):
                span = [0, 0]
                spans.append(span)

            if len(ys) == 1 and ys[0].startswith('I'):
                ys[0] = 'B' + ys[0][1:]
                span = [0, 0]
                spans.append(span)

            while end < len(ys):

                span = []
                if end < len(ys) and ys[start].startswith('O'):
                    # 应对O的情况
                    start += 1
                    end += 1

                # TODO 情况123可以看成B开头，合并成一个if判断，内部再先判断ys[end]是否存在（单B），不存在时是I（BI）或者不是I（BO或者BB）
                # 情况1，BO、BB
                if end < len(ys) and ys[start].startswith('B') and not ys[end].startswith('I'):
                    # 应对BB或者BO的情况
                    span = [start, start]
                    spans.append(span)
                    start += 1
                    end += 1

                # 情况2，BI
                if end < len(ys) and ys[start].startswith('B') and ys[end].startswith('I'):
                    while end < len(ys) and ys[end].startswith('I'):
                        # and ys[start].split('-')[1] == ys[end].split('-')[1]
                        end += 1

                    span = [start, end - 1]
                    spans.append(span)
                    start = end
                    end += 1

                # 情况3，B 在最后
                if start == len(ys) - 1 and ys[start].startswith('B'):
                    span = [start, start]
                    spans.append(span)
                    start += 1
                    end += 1

                # 情况4，OI开头，伪标签可能会存在这种情况，需要将其改为B
                if end < len(ys) and ys[start].startswith('I'):
                    ys[start] = 'B' + ys[start][1:]

            entity_num = 0
            for d in ys:
                if d.startswith('B'):
                    entity_num += 1

            assert entity_num == len(spans), "entity数量与spans的长度不一致！"
            ys = raw_ys.split()
            data.append((xs, ys, spans, domain))
        return data


class ATISProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ['arrive_date.month_name', 'depart_date.date_relative', 'stoploc.airport_name',
                       'fromloc.city_name', 'class_type', 'flight_stop', 'day_name', 'depart_date.today_relative',
                       'fromloc.state_code', 'stoploc.state_code', 'fromloc.airport_code', 'flight_mod',
                       'arrive_time.start_time', 'toloc.city_name', 'arrive_time.time', 'arrive_date.today_relative',
                       'depart_time.start_time', 'depart_time.end_time', 'arrive_date.date_relative',
                       'return_date.day_name', 'time_relative', 'state_code', 'stoploc.airport_code', 'booking_class',
                       'days_code', 'flight', 'depart_date.day_name', 'return_date.today_relative', 'meal',
                       'return_date.day_number', 'arrive_date.day_number', 'or', 'transport_type', 'toloc.country_name',
                       'depart_time.period_of_day', 'arrive_time.end_time', 'meal_code', 'airline_code', 'round_trip',
                       'day_number', 'depart_date.month_name', 'return_time.period_of_day', 'time', 'state_name',
                       'depart_time.period_mod', 'fromloc.airport_name', 'flight_days', 'toloc.state_name',
                       'depart_time.time', 'toloc.airport_name', 'airport_code', 'connect', 'economy',
                       'stoploc.city_name', 'period_of_day', 'toloc.state_code', 'mod', 'toloc.airport_code',
                       'arrive_time.period_of_day', 'compartment', 'cost_relative', 'arrive_time.time_relative',
                       'month_name', 'depart_date.year', 'arrive_date.day_name', 'city_name', 'fare_basis_code',
                       'meal_description', 'aircraft_code', 'return_time.period_mod', 'fromloc.state_name',
                       'airport_name', 'flight_number', 'restriction_code', 'depart_date.day_number',
                       'arrive_time.period_mod', 'fare_amount', 'airline_name', 'flight_time', 'today_relative',
                       'depart_time.time_relative', 'return_date.month_name', 'return_date.date_relative', 'O']

    def get_examples(self, data_dir, split=None, domain=None, use_true_bio_label=False):
        assert domain is not None
        path = os.path.join(data_dir, "{}/{}.tsv".format(domain, split))
        # path = data_dir
        with open(path, encoding='utf8') as f:
            data = SNIPSProcessor.load_data(f, split, use_true_bio_label)
            examples = []
            for idx, (xs, ys, spans, domain) in enumerate(data):
                for span in spans:
                    text_a = " ".join(xs)
                    # 判断是否为train
                    if split == 'train':
                        # slot = ys[span[0]].split('-')[1]
                        meta = {
                            "entity": " ".join(xs[span[0]: span[1] + 1]),
                            "domain": dataset2domain_desp['ATIS'].get(domain, 'others'),
                            }
                    else:
                        meta = {
                            "entity": " ".join(xs[span[0]: span[1] + 1]),
                            "domain": dataset2domain_desp['ATIS'].get(domain, 'others'),
                            }
                    # 对于验证集与测试集，如果span对应的label为O时，则对应的span label也为O，如果不为O，那么去掉B-或I-前缀赋给span label
                    if ys[span[0]].startswith('O'):
                        span_label = self.get_label_id('O')
                    else:
                        span_label = self.get_label_id(ys[span[0]][2:])

                    #  将整个句子的原始标签ys以及跨度信息spans传进batch中， 简单点可以一起放进label里面传递
                    label = {"ys": ys, "span": span, "span_label": span_label}
                    # ys 是 list[int], span 也是list[int]， label是int
                    example = InputExample(guid=str(idx), text_a=text_a, meta=meta, label=label)
                    examples.append(example)

            return examples


class MultiWOZProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ['hotel-parking', 'restaurant-name', 'hotel-bookstay', 'taxi-destination', 'train-bookpeople',
                       'hotel-stars', 'restaurant-area', 'hotel-area', 'train-destination', 'hotel-internet',
                       'restaurant-bookday', 'restaurant-food', 'restaurant-booktime', 'hotel-bookpeople',
                       'hotel-bookday', 'train-departure', 'attraction-type', 'hotel-type', 'hospital-department',
                       'train-leaveat', 'bus-destination', 'restaurant-bookpeople', 'bus-departure', 'train-day',
                       'hotel-name', 'taxi-leaveat', 'taxi-arriveby', 'attraction-area', 'restaurant-pricerange',
                       'attraction-name', 'taxi-departure', 'train-arriveby', 'hotel-pricerange', 'O']

    def get_examples(self, data_dir, split=None, domain=None, use_true_bio_label=False):
        assert domain is not None
        path = os.path.join(data_dir, "{}/{}.tsv".format(domain, split))
        # path = data_dir
        with open(path, encoding='utf8') as f:
            data = SNIPSProcessor.load_data(f, split, use_true_bio_label)
            examples = []
            # TODO 需要考虑如何添加eg，不能直接添加同entity类别的eg，因为评估的时候看不到
            for idx, (xs, ys, spans, domain) in enumerate(data):
                for span in spans:
                    text_a = " ".join(xs)
                    # 判断是否为train
                    if split == 'train':
                        meta = {
                            "entity": " ".join(xs[span[0]: span[1] + 1]),
                            "domain": dataset2domain_desp['MultiWOZ'].get(domain, 'others'),
                            }
                    else:
                        meta = {
                            "entity": " ".join(xs[span[0]: span[1] + 1]),
                            "domain": dataset2domain_desp['MultiWOZ'].get(domain, 'others'),
                            }
                    # 对于验证集与测试集，如果span对应的label为O时，则对应的span label也为O，如果不为O，那么去掉B-或I-前缀赋给span label
                    if ys[span[0]].startswith('O'):
                        span_label = self.get_label_id('O')
                    else:
                        span_label = self.get_label_id(ys[span[0]][2:])

                    #  将整个句子的原始标签ys以及跨度信息spans传进batch中， 简单点可以一起放进label里面传递
                    label = {"ys": ys, "span": span, "span_label": span_label}
                    # ys 是 list[int], span 也是list[int]， label是int
                    example = InputExample(guid=str(idx), text_a=text_a, meta=meta, label=label)
                    examples.append(example)

            return examples


class SGDProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ['time', 'movie_title', 'price_per_night', 'check_out_date', 'destination', 'balance', 'fare',
                       'to_station', 'outbound_departure_time', 'place_name', 'city', 'alarm_name', 'stylist_name',
                       'pickup_city', 'date_of_journey', 'amount', 'destination_airport', 'return_date',
                       'venue_address', 'street_address', 'stay_length', 'start_date', 'starring', 'actors', 'rating',
                       'attraction_name', 'event_location', 'rent', 'doctor_name', 'precipitation', 'therapist_name',
                       'average_rating', 'cuisine', 'leaving_time', 'to_location', 'price_per_day', 'area',
                       'approximate_ride_duration', 'date', 'origin_city', 'to_city', 'phone_number', 'total',
                       'show_time', 'journey_start_time', 'percent_rating', 'hotel_name', 'event_date', 'show_date',
                       'origin_airport', 'transfer_time', 'appointment_date', 'origin_airport_name', 'directed_by',
                       'available_end_time', 'from_location', 'wait_time', 'dentist_name', 'from_city', 'title',
                       'end_date', 'departure_time', 'departure_date', 'wind', 'price_per_ticket',
                       'destination_airport_name', 'genre', 'track', 'aggregate_rating', 'check_in_date', 'pickup_date',
                       'recipient_account_name', 'transfer_amount', 'property_name', 'new_alarm_time', 'director',
                       'cast', 'subcategory', 'destination_city', 'to', 'visit_date', 'outbound_arrival_time',
                       'origin_station_name', 'address_of_location', 'artist', 'event_time', 'pickup_time',
                       'new_alarm_name', 'temperature', 'origin', 'receiver', 'venue', 'inbound_departure_time',
                       'from_station', 'album', 'event_name', 'ride_fare', 'recipient_name', 'inbound_arrival_time',
                       'leaving_date', 'account_balance', 'from', 'song_name', 'address', 'theater_name', 'price',
                       'restaurant_name', 'contact_name', 'available_start_time', 'car_name', 'location',
                       'appointment_time', 'number_of_days', 'category', 'pickup_location', 'where_to', 'total_price',
                       'dropoff_date', 'humidity', 'destination_station_name', 'alarm_time', 'city_of_event',
                       'movie_name', 'O']

    def get_examples(self, data_dir, split=None, domain=None, use_true_bio_label=False):
        # 这里的domain为str，形式为AddToPlaylist_0等
        assert domain is not None
        path = os.path.join(data_dir, "{}/{}.tsv".format(domain, split))
        # path = data_dir
        with open(path, encoding='utf8') as f:
            data = SNIPSProcessor.load_data(f, split, use_true_bio_label)
            examples = []
            for idx, (xs, ys, spans, domain) in enumerate(data):

                for span in spans:
                    text_a = " ".join(xs)
                    # 判断是否为train
                    if split == 'train':
                        meta = {
                            "entity": " ".join(xs[span[0]: span[1] + 1]),
                            "domain": dataset2domain_desp['SGD'].get(domain, 'others'),
                            }
                    else:

                        meta = {
                            "entity": " ".join(xs[span[0]: span[1] + 1]),
                            "domain": dataset2domain_desp['SGD'].get(domain, 'others'),
                            }
                    # 对于验证集与测试集，如果span对应的label为O时，则对应的span label也为O，如果不为O，那么去掉B-或I-前缀赋给span label
                    if ys[span[0]].startswith('O'):
                        span_label = self.get_label_id('O')
                    else:
                        span_label = self.get_label_id(ys[span[0]][2:])

                    #  将整个句子的原始标签ys以及跨度信息spans传进batch中， 简单点可以一起放进label里面传递
                    label = {"ys": ys, "span": span, "span_label": span_label}
                    # ys 是 list[int], span 也是list[int]， label是int
                    example = InputExample(guid=str(idx), text_a=text_a, meta=meta, label=label)
                    examples.append(example)

            return examples


PROCESSORS = {
    "ATIS": ATISProcessor, "MultiWOZ": MultiWOZProcessor, "SGD": SGDProcessor, "SNIPS": SNIPSProcessor,
    }
