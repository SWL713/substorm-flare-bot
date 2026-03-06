import requests
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime

# NOAA flare feed
url = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"

data = requests.get(url).json()

flare = data[-1]

flare_class = flare["max_class"]
start = flare["begin_time"][11:16]
peak = flare["max_time"][11:16]
end = flare["end_time"][11:16]

duration = (
    datetime.fromisoformat(flare["end_time"].replace("Z","")) -
    datetime.fromisoformat(flare["begin_time"].replace("Z",""))
).seconds // 60

template = Image.open("template.png")
draw = ImageDraw.Draw(template)

font_big = ImageFont.truetype("Arial.ttf",120)
font_small = ImageFont.truetype("Arial.ttf",40)

draw.text((540,550),flare_class,anchor="mm",fill=(255,140,0),font=font_big)

draw.text((350,700),f"Start : {start} UTC",fill="white",font=font_small)
draw.text((700,700),f"Peak : {peak} UTC",fill="white",font=font_small)
draw.text((350,760),f"End : {end} UTC",fill="white",font=font_small)
draw.text((700,760),f"Duration : {duration} minutes",fill="white",font=font_small)

filename = f"cards/flare_{flare_class}_{peak}.png"

template.save(filename)