
import json
import os
import hashlib

USER_DATA_FILE = 'users.json'

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def signup(username, password, role='user'):
    users = load_users()
    if username in users:
        return False
    users[username] = {'password': hash_password(password), 'role': role}
    save_users(users)
    return True

def login(username, password):
    users = load_users()
    user = users.get(username)
    if user and user['password'] == hash_password(password):
        return user['role']
    return None
