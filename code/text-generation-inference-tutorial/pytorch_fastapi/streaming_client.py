import requests
import sys
import json

url = "http://localhost:8080/generate_stream"


def inference(line):
  d = {
    "prompt": line,
    "max_length": 100,
    "min_length": 20, 
    "repetition_penalty": 10,
    "do_sample": False,
    "top_k": 10,
    "top_p": 0.9,
    "temperature": 0.9
  }
  response = requests.post(
      url,
      stream=True,
      headers={"accept": "application/json"},
      data= json.dumps(d)
  )

  for chunk in response.iter_content(chunk_size=64):
      if chunk:
          print(str(chunk, encoding="utf-8"), end="")
          sys.stdout.flush()

# read intput from stdin
def run():
  while True:
    print("\n####Prompts:",)
    line = sys.stdin.readline()
    if line == 'exit':
        break
    print("\n####Answer: ")
    inference(line) 

if __name__ == '__main__':
  run()