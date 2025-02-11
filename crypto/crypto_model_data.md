I have divided the crypto PCAP into train/test pcaps
Then the test pcap file has been merged with Solana pcaps. So that the final test pcap has both Mining and other application data. 


Do the below tasks, 

Task1: 
=======
1. Take data from dVC macine 
  scp solana@192.168.10.211:/data/Solana_datasets/mn-data/external_datasets/bitcoin/model_data/final_data .
2. You will get final train and test pcap files. 

Task 2.
========
With the already trained model, use the test file to evaluate the model. 
On my side I got below resutls, check on your side whether you get the same results
Examin the data were traffic_type is "Unkown"

===================
>>> df.groupby('traffic_type').size()
traffic_type
Audio Chat         7
File Transfer      1
Mining           283
P2P                1
Streaming         65
Text Chat         20
Unknown          302
Video Chat         1

=================
>>> df.groupby('application').size()
application
Adobe Ads               1
Amazon                 37
Canadianshield         54
Cloudflare             10
Cookielaw               6
Dailymotion             1
Deezer                  2
DoubleClick             4
Facebook Messenger      1
Fastly                 14
Google                 71
Google Analytics        4
Google Chat             2
Line                    2
Meta                    1
Microsoft Teams         1
Mozilla                 6
Netflix                69
OneTrust                2
Pinterest               4
Pippio                  2
Scorecard               2
Sentry                  1
Signal                  1
Spotify                24
Telegram                3
Twitter                 3
Ubuntu                  1
Unknown               324
Whatsapp                1
Youtube                16
Zoom                   10

Task - 3 
===========
re-train the model with Train file. 
Then do the evaluation using the test data. 



