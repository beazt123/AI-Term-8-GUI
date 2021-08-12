import re
import numpy as np
import streamlit as st
import torch
from model import model
from lib import nlp
from joblib import load


UNIQUE_DATES = load("listOfDates.joblib")
WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

def extract_special_words(s, char = "#"):
    return [part[1:] for part in s.split() if part.startswith(char) and part != char]

def findURLs(string):
    return re.findall(WEB_URL_REGEX, string)




st.write("""
# COVID19 Retweet Prediction
Fill out the blanks below to predict how many retweets your tweet will get!
""")

# st.button('Hit me')
# st.checkbox('Check me out')
# st.radio('Radio', [1,2,3])
# st.selectbox('Select', [1,2,3])
# st.multiselect('Multiselect', [1,2,3])
# st.slider('Slide me', min_value=0, max_value=10)
# st.select_slider('Slide to select', options=[1,'2'])
# st.text_input('Enter some text')
# st.number_input('Enter a number')
# st.text_area('Area for textual entry')
# st.date_input('Date input')
# st.time_input('Time entry')
# st.file_uploader('File uploader')
# st.color_picker('Pick a color')

# st.text('Fixed width text')
# st.markdown('_Markdown_') # see *
# st.latex(r''' e^{i\pi} + 1 = 0 ''')
# st.write('Most objects') # df, err, func, keras!
# st.write(['st', 'is <', 3]) # see *
# st.title('My title')
# st.header('My header')
# st.subheader('My sub')
# st.code('for i in range(8): foo()')

with st.form("my_form"):
    st.markdown("## Tweet Details")
    tweet = st.text_area('Tweet!')
    st.markdown('For the next 2 blanks, consult this [link](http://sentistrength.wlv.ac.uk/TensiStrength.html) to check the positive & negative sentiments')
    positive_sentiment = st.number_input('SentiStrength Positive sentiment', step = 1)
    negative_sentiment = st.number_input('SentiStrength Negative sentiment', step = 1)
    followers = st.number_input('How many followers do u have? Are u famous? Or are you an influencer?', step = 1)
    friends = st.number_input('How many friends do u have on Twitter?', step = 1)
    favorites = st.number_input('How many likes did u have for your tweet?', step = 1)

    tweet_date = st.date_input('Date of tweet')
    # print(type(tweet_date))
    tweet_time = st.time_input('Time of tweet')
    # print(type(tweet_time))
    day_of_year = int(tweet_date.strftime('%j'))
    day_of_week = tweet_date.weekday()
    hour_of_day = tweet_time.hour

    sine_hour = np.sin((hour_of_day / 23) * 2 * np.pi)
    cosine_hour = np.cos((hour_of_day / 23) * 2 * np.pi)
    sine_day = np.sin((day_of_year / 365) * 2 * np.pi)
    cosine_day = np.cos((day_of_year / 365) * 2 * np.pi)

    sine_day_of_week = np.sin((day_of_week / 6) * 2 * np.pi) # currently assumes that day of week 0-6 for monday to sunday
    cosine_day_of_week = np.cos((day_of_week / 6) * 2 * np.pi)
    weekend = day_of_week == 5 or day_of_week == 6


    # "entity_count",
    entities = [(ent.text, ent.label_) for ent in nlp(tweet).ents]
    # print(entities)
    entity_count = len(entities)
    # print(f"detected entities: {entities}")

    # "hashtag_count",
    hashtags = extract_special_words(tweet, "#")
    hashtag_count = len(hashtags)
    # print(f"detected hashtags: {hashtags}")

    # "mention_count",
    mentions = extract_special_words(tweet, "@") # https://stackoverflow.com/questions/2527892/parsing-a-tweet-to-extract-hashtags-into-an-array
    mention_count = len(mentions)
    # print(f"detected mentions: {mentions}")

    # "url_count",
    urls = findURLs(tweet) # https://stackoverflow.com/questions/9760588/how-do-you-extract-a-url-from-a-string-using-python
    url_count = len(urls)
    # print(f"detected urls: {urls}")

    # "tlen",
    tlen = hashtag_count + mention_count + url_count + entity_count

    # "ratio_fav_#followers",
    ratio_fav_followers = favorites / (followers + 1)


    # "sentiment_ppn",
    sentiment_ppn = positive_sentiment + negative_sentiment

    # "time_importance"
    try:
        time_importance = UNIQUE_DATES.index(tweet_date)
    except ValueError:
        time_importance = 0

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        x = [
            followers,
            friends,
            favorites,
            weekend,
            entity_count,
            hashtag_count,
            mention_count,
            url_count,
            tlen,
            ratio_fav_followers,
            time_importance,
            sentiment_ppn,
            sine_hour,
            cosine_hour,
            sine_day,
            cosine_day,
            sine_day_of_week,
            cosine_day_of_week
        ]
        X = torch.Tensor(x)     
        # print(X)   
        # print(X.size())
        # print(X.view(-1,1).size())

        logRetweets = model(X.view(-1,1).T)
        numRetweets = np.exp(logRetweets.item())
        st.markdown(f'Number of retweets is:\t\t{numRetweets}')

