import pandas as pd
import numpy as np
import requests
import os
import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)



class Util:
    def __init__(self, queue_path, webhook_url, server_name, datetime_format = "%y/%m/%d %H:%M:%S"):
        self.queue_path = queue_path
        self.webhook_url = webhook_url
        self.server_name = server_name
        self.datetime_format = datetime_format
        self.now = datetime.datetime.now

    def send_slack_notification(self, message):
        slack_data = {'text': message}
        
        try:
            response = requests.post(self.webhook_url, json=slack_data)
        except:
            print("Fail to send a slack message")

        if response.status_code != 200:
            print(f'Request to Slack returned an error {response.status_code}, the response is:\n{response.text}')
    
    def read_queue(self):
        try:
            self.queue = pd.read_csv(self.queue_path, sep = ",", dtype = str)
            self.queue = self.queue.replace(r'^\s*$', "", regex = True) # 스페이스바만 있으면 "" 값으로 변경
            self.queue = self.queue.replace(np.nan, "", regex = True) # nan값이면 ""로 변경
        except:
            pass

        return self.queue

    def write_queue(self):
        try:
            self.queue.to_csv(self.queue_path, sep = ",", na_rep = "", index = False)
        except:
            pass

    def update_queue_from_current(self, current):
        self.read_queue()
        idx1 = self.queue["file_path"] == current["file_path"]
        idx2 = self.queue["argument"] == current["argument"]
        idx = self.queue.index[idx1 & idx2].tolist()

        if len(idx):
            self.queue.loc[idx, ["start"]] = current["start"]
            self.queue.loc[idx, ["end"]] = current["end"]
        else: # 원래 queue에 있던 current가 사라졌으면 current 값을 queue 맨 아래에 추가
            self.queue = pd.concat([self.queue, current])
        
        self.write_queue()
            
        
    def get_current_exec(self):
        self.read_queue()
        index = self.queue.index[self.queue["end"] == ""].tolist()

        if (len(index)):
            current = self.queue.loc[index[0]]
            return current, len(index)       
        else:
            return None, None


    def get_current_time(self):
        time = self.now()
        time_str = time.strftime(self.datetime_format)
        return time, time_str

    def log(self, msg, time_str = None):
        if time_str is None:
            _, time_str = self.get_current_time()
        
        new_msg = f"{time_str} - {self.server_name}: {msg}"
        # self.send_slack_notification(new_msg)
        print(new_msg)

