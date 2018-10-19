import argparse
import random
import time
import uuid
import requests
import json

import rl_client

class my_error_callback(rl_client.error_callback):
  def on_error(self, error_code, error_message):
    print("Background error:")
    print(error_message)

def load_config_from_json(file_name):
    with open(file_name, 'r') as config_file:
        return rl_client.create_config_from_json(config_file.read())

class person:
    def __init__(self, id, major, hobby, fav_char, prob):
        self._id = id
        self._major = major
        self._hobby = hobby
        self._favorite_character = fav_char
        self._topic_click_probability = prob

    def get_features(self):
        return '"User":{{"id":"{}","major":"{}","hobby":"{}","favorite_character":"{}"}}'.format(self._id, self._major, self._hobby, self._favorite_character)

    def get_outcome(self, chosen_action):
        norm_draw_val = random.random()
        click_prob = self._topic_click_probability[chosen_action]
        if (norm_draw_val <= click_prob):
            return 1.0
        else:
            return 0.0

class rl_sim:
    def __init__(self, args):
        self._options = args

        self.config = load_config_from_json(self._options.json_config)
        self._rl_client = rl_client.live_model(self.config, my_error_callback())
        self._rl_client.init()

        tp1 = {'HerbGarden': 0.3, "MachineLearning": 0.2, "Tennis": 0.5 }
        tp2 = {'HerbGarden': 0.1, "MachineLearning": 0.4, "Tennis": 0.01 }

        self._actions = ['HerbGarden', 'MachineLearning', 'Tennis']
        self._people = [
            person('rnc', 'engineering', 'hiking', 'spock', tp1),
            person('mk', 'psychology', 'kids', '7of9', tp2)]
            
        # REST-API settings
        if self._options.rew_via_rest:
            appId = json.load(open(self._options.json_config))["ApplicationID"]
            self.rest_url = 'https://ds.microsoft.com/api/v2/'+appId+'/reward/'
            self.rest_site = 'site://'+appId
            self.rest_session = requests.Session()
            self.rest_session.headers.update({'Content-Type': 'application/json', 'Accept':'application/json'})

    def loop(self):
        round = 0
        stats = {person._id : [[0,0] for _ in self._actions] for person in self._people}
        ctr = [0,0]
        while round != self._options.num_rounds:
            try:
                round += 1

                # random.sample() returns a list
                person = random.sample(self._people, 1)[0]
                
                # create context
                shared_features = person.get_features()
                action_features = '"_multi":[' + ','.join('{"a":{"topic":"'+action+'"}}' for action in self._actions) + ']'
                context = '{' + shared_features + ',' + action_features + '}'
                
                event_id = str(uuid.uuid4())+'-'

                model_id, chosen_action_id, actions_probabilities = self._rl_client.choose_rank(event_id, context, deferred=self._options.do_deferred)

                self._rl_client.report_action_taken(event_id)
                
                stats[person._id][chosen_action_id][1] += 1
                ctr[1] += 1

                outcome = person.get_outcome(self._actions[chosen_action_id])
                if outcome != 0:
                    if self._options.rew_via_rest:
                        r2 = self.rest_session.post(self.rest_url+event_id, json=outcome)
                        if r2.status_code != 200 or r2.headers.get('x-msdecision-src', None) != self.rest_site:
                            err[1] += 1
                            print('Reward Error - status_code: {}; headers: {}'.format(r2.status_code,r2.headers))
                    else:
                        self._rl_client.report_outcome(event_id, outcome)
                    ctr[0] += 1
                    stats[person._id][chosen_action_id][0] += 1

                if round % 1000 == 0:
                    print('Round: {}, ctr: {:.1%}'.format(round, ctr[0]/ctr[1]), stats, model_id, self._options.do_deferred)

                time.sleep(0.001)
            except Exception as e:
                print(e)
                time.sleep(2)
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_config', help='client json config', required=True)
    parser.add_argument('--num_rounds', help='number of rounds (default: infinity)', type=int, default=-1)
    parser.add_argument('--rew_via_rest', help='Send rewards via REST-API (default: false)', action='store_true')
    parser.add_argument('--do_deferred', help='Enable deferred actions (default: false)', action='store_true')

    vm = parser.parse_args()
    sim = rl_sim(vm)
    sim.loop()
