from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    validation_token = request.args.get('validationtoken')
    print(validation_token)
    print(request.args)
    if validation_token:
        print("returning v",validation_token)
        return validation_token, 200
    # Handle other webhook events here
    print("returning success status")
    return jsonify({'status': 'received'}), 200

if __name__ == '__main__':
    app.run(debug=True)