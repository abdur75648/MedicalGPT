import os
from tqdm import tqdm
from openai import OpenAI
import base64 
import json
openaikey = os.environ.get("OPENAI_API_KEY")

# OpenAI API client
client = OpenAI(
    api_key=openaikey) # OpenAI API key

with open('indiana_filtered.json', 'r') as f:
    data = json.load(f)

data = data[:10]
print("Number of samples being processed: ", len(data))

# Function to encode image to base64
def encode_image(image_path):
  with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

# Function to generate the response
def llm_output (system_prompt_content, user_prompt_content, image_path):
  base64_image = encode_image(image_path)
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
          {
          "role": "system",
          "content": system_prompt_content
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": user_prompt_content },
          {
            "type": "image_url",
            "image_url": {
            "url":  f"data:image/jpeg;base64,{base64_image}",

            },
          },
        ],
      }
    ],
    temperature = 0,
    max_tokens=1000,
  ) 
  return response.choices[0].message.content


vqa_data = []
for item in tqdm(data):
    filename = item['filename']
    item.pop('filename')
    
    # Convert the record to a string
    item = json.dumps(item)
    
    ############################ Report Generation and VQA Dataset Format Conversion ############################
    print("Generating a report from the image and JSON data")
    # System prompt for detailed analysis and report generation
    system_prompt_content_for_report_generation = "You are an AI assistant with advanced capabilities in radiology. \
                            Your task is to analyze the attached medical image and the associated information from a JSON file. \
                            Based on your analysis, you need to prepare a detailed report of the image, explaining \
                            everything - What the image is of, what is the modality, which body part, and what are the findings, problems,, abnormalities, or \
                            any other relevant information. It should be a comprehensive report that can be used by a medical professional. Your response should be plain text format."
    
    # User prompt for detailed analysis and report generation
    user_prompt_content_report_generation = "Please process this image attached. The information I have from its JSON file is this - " + item + ". I need you to analyze an image and prepare a detailed report out of it using both your findings and from the JSON data. \
                        It should be a few paragraphs long (Plain text format, good quality language), explaining \
                        everything - What the image is of, what is the modality, which body part, and what are the findings, problems,, abnormalities, or \
                        any other relevant information."
    
    # Generate the response (Report generation)
    response = llm_output(system_prompt_content_for_report_generation, user_prompt_content_report_generation, filename)
    # print(response)
    
    # System prompt for VQA dataset format conversion                        
    system_prompt_content_for_VQA_format = "You are an AI assistant with advanced capabilities in formatting responses for VQA datasets. \
                            Your task is to convert the given plain text response into a VQA dataset format. \
                            You need to create a dictionary containing two keys: 'prompt' and 'response'. \
                            The 'response' key should contain the given plain text response as it is. \
                            The 'prompt' key should be a well-crafted prompt that asks the model to generate a detailed report of the image. \
                            Example Response: '\
                            {'prompt': 'Describe the given image in detail in a report format or something like What are the key findings in this attached xray/ct/MRI scan? Prepare a detailed report of the image.', \
                                'response': Given plain text response}'"
    
    # User prompt for VQA dataset format conversion
    user_prompt_content_VQA_format = "Please convert this plain text response into a VQA dataset format. The plain text response is - " + response + ". You need to create a dictionary containing two keys: 'prompt' and 'response'. \
                        The 'response' key should contain the given plain text response as it is, just remove invalid characters. \
                        The 'prompt' key should be a well-crafted prompt that asks the model to generate a detailed report of the image."
    # print(user_prompt_content_VQA_format)
    
    # Generate the response (VQA dataset format conversion)
    response_vqa = llm_output(system_prompt_content_for_VQA_format, user_prompt_content_VQA_format, filename)
    # ```json
    # {
    # "prompt": "Describe the given chest X-ray image in detail in a report format, including any significant findings and impressions.",
    # "response": "The provided image is a frontal chest X-ray, a common radiographic modality used to evaluate the thoracic cavity, including the lungs, heart, and surrounding structures. The image is a posteroanterior (PA) view, which is typically preferred for its ability to provide a clearer view of the lungs and heart with less magnification of the heart shadow.\n\nUpon analyzing the image and correlating it with the provided JSON data, several significant findings are noted:\n\n1. **Diffuse Bilateral Interstitial and Alveolar Opacities**: The image shows widespread interstitial and alveolar opacities in both lungs. These findings are consistent with chronic obstructive pulmonary disease (COPD) and bullous emphysema. The interstitial opacities suggest the presence of interstitial fibrosis, which is a hallmark of chronic lung disease.\n\n2. **Bullous Emphysema**: There are areas of hyperlucency, particularly in the upper lobes, indicative of bullous changes. Bullous emphysema is characterized by the presence of large air-filled spaces (bullae) within the lung parenchyma, resulting from the destruction of alveolar walls.\n\n3. **Irregular Opacities in the Left Lung Apex**: The left lung apex shows irregular opacities, which could represent a cavitary lesion. This finding raises the possibility of a cavitary process, such as tuberculosis or a fungal infection, although scarring is also a consideration.\n\n4. **Streaky Opacities in the Right Upper Lobe**: The right upper lobe exhibits streaky opacities, which may be indicative of scarring. This could be a result of previous infections, inflammation, or other chronic lung conditions.\n\n5. **Cardiomediastinal Silhouette**: The cardiomediastinal silhouette appears normal in size and contour, suggesting that there is no significant cardiomegaly or mediastinal shift.\n\n6. **Absence of Pneumothorax or Large Pleural Effusion**: There is no evidence of pneumothorax (air in the pleural space) or large pleural effusion (fluid in the pleural space), which are important considerations in the differential diagnosis of chest pain and respiratory distress.\n\n**Impression**:\n1. The findings are consistent with bullous emphysema and interstitial fibrosis.\n2. The irregular opacities in the left lung apex are likely due to scarring, but a cavitary lesion cannot be excluded without further imaging.\n3. The bilateral upper lobe opacities could represent scarring. Given the absence of a comparison exam, a short interval follow-up radiograph or a CT scan of the thorax is recommended to document resolution or further characterize these findings.\n\nIn summary, the chest X-ray reveals significant chronic lung changes, including bullous emphysema and interstitial fibrosis, with additional findings that warrant further investigation to rule out active or progressive disease processes."
    # }
    # ```
    
    # Convert the response to a dictionary after removing the JSON formatting
    response_vqa = response_vqa.replace('```json', '').replace('```', '')
    response_vqa = json.loads(response_vqa)
    # print(response_vqa)
    
    # if it's a dict, append it to the list
    if type(response_vqa) == dict:
        response_vqa["image"] = filename
        # Replace the key name response with caption
        response_vqa["caption"] = response_vqa.pop("response")
        vqa_data.append(response_vqa)
    else:
        print("Response is not a dictionary")
        continue


    ############################ VQA Only From OpenI Dataset ############################
    print("Generating a VQA pair from the image and JSON data")
    # System prompt for generating any good quality question and its answer from the image and its JSON data
    system_prompt_content_for_question_answer_pair_generation = "You are an AI assistant with advanced capabilities in radiology. \
                            Your task is to analyze the attached medical image and the associated information from a JSON file. \
                            Based on your analysis, you need to prepare a good quality visual question and its answer from the image and the JSON data. \
                            The question should be relevant to the image and the answer should be a detailed explanation of the question. \
                            Return it in the form of a dictionary containing two keys: 'prompt' and 'response'"
    
    # User prompt for generating any good quality question and its answer from the image and its JSON data
    user_prompt_content_question_answer_pair_generation = "Please process this image attached. The information I have from its JSON file is this - " + item + ". I need you to analyze the image, combine knowledge from the JSON data, and generate a good quality visual question and its answer. \
                        The question should be relevant to the image and the answer should be a detailed explanation of the question. They should be direct, concise and informative. \
                        Return it in the form of a dictionary containing two keys: 'prompt' and 'response'."
                        
    # Generate the response (VQA dataset directly from the image and JSON data)
    response_vqa = llm_output(system_prompt_content_for_question_answer_pair_generation, user_prompt_content_question_answer_pair_generation, filename)
    
    # Convert the response to a dictionary after removing the JSON formatting
    response_vqa = response_vqa.replace('```json', '').replace('```', '')
    response_vqa = json.loads(response_vqa)
    # print(response_vqa)
    
    # if it's a dict, append it to the list
    if type(response_vqa) == dict:
        response_vqa["image"] = filename
        # Replace the key name response with caption
        response_vqa["caption"] = response_vqa.pop("response")
        vqa_data.append(response_vqa)
    else:
        print("Response is not a dictionary")
        continue
    
print(vqa_data)

# Randomly shuffle the VQA data
import random
random.shuffle(vqa_data)

# Save the VQA data in a new JSON file
with open('indiana_vqa.json', 'w') as f:
    json.dump(vqa_data, f, indent=2)
    