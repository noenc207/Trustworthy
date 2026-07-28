import urllib.request
import re
req = urllib.request.Request('https://data.mendeley.com/datasets/zr7vgbcyr2/1', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
print(re.findall(r'https://[^\s\"\'\>]+', html))
