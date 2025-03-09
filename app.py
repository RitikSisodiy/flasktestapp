from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    validation_token = request.args.get('validationtoken')
    print("Query Parameters:", request.args)

    # Print raw request data
    print("Raw Data:", request.data)

    # Print JSON data if available
    if request.is_json:
        print("JSON Data:", request.json)

    # Print form data if applicable
    print("Form Data:", request.form)
    if validation_token:
        print("returning v",validation_token)
        return validation_token, 200
    # Handle other webhook events here
    print("returning success status")
    return jsonify({'status': 'received'}), 200

if __name__ == '__main__':
    app.run(debug=True)