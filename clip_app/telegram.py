#!/usr/bin/env python3 

from flask import Flask, request, jsonify
import argparse
import requests
import os
import json
import configparser

app = Flask(__name__)

INI_PATH = os.path.join(os.path.dirname(__file__), 'telegram.ini')

def read_ids_from_ini(file_path, section, option):
    """
    Reads a list of IDs from a specified section and option in an INI file.

    Args:
        file_path (str): Path to the INI file.
        section (str): Section in the INI file.
        option (str): Option under the section.

    Returns:
        list: List of IDs as strings.
    """
    config = configparser.ConfigParser()
    config.read(file_path)

    if config.has_section(section) and config.has_option(section, option) :
        if section == 'IDs':
            ids = config.get(section, option)
            return [id.strip() for id in ids.split(",")]
        else:
            return config.get(section, option)
    else:
        raise ValueError(f"Section '{section}' or option '{option}' not found in the INI file.")

def get_ids():
    TOKEN = read_ids_from_ini(INI_PATH, "BOT", "token")
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    resp = requests.get(url).json() 

    print(f'Listing all Users in Telegram bot bot{TOKEN}')
    for i, res in enumerate(resp['result']):
        print('\t{i}: Name = {name}, ID = {id}'.format(i=i,
            name = res['message']['chat']['first_name'],
            id = res['message']['chat']['id']))


def send_telegram_message(message, chat_ids, BOT_TOKEN):

    TELEGRAM_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    print(f'chat_ids={chat_ids}')

    for c_id in chat_ids:
        print(f'chat_id={c_id}')
        payload = {
            "chat_id": c_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            print(f"Sending request to: {TELEGRAM_URL}")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(TELEGRAM_URL, json=payload)
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            response.raise_for_status()
            print("Message sent successfully")

        except requests.exceptions.RequestException as e:
            return False, f"Failed to send message: {str(e)}\nResponse: {response.text if 'response' in locals() else 'No response'}"

    return True, "All Messages sent successfully"
 

@app.route('/notify', methods=['POST'])
def notify():
    chat_ids = read_ids_from_ini(INI_PATH, "IDs", "list")
    BOT_TOKEN = read_ids_from_ini(INI_PATH, "BOT", "token")
        
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    success, message = send_telegram_message(data['message'], chat_ids, BOT_TOKEN)
    if success:
        return jsonify({'status': 'success', 'message': message})
    return jsonify({'status': 'error', 'message': message}), 500

def set_args_parser(parser):
    parser.description = "A python script for sending Telegram messages to bots"
    parser.add_argument('-c', '--chat-id', help='Optional flag to get a list of chat IDs.', action='store_true')

def main():
    app.run(host='0.0.0.0', port=5001, debug=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    set_args_parser(parser)
    args = parser.parse_args()
    if args.chat_id:
        get_ids()
    else:
        main()
    
