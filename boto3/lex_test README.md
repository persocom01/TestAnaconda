# inp-lex-tester

Code for testing of amazon lex chatbots.

## Installation

This program requires the following python modules:

* boto3 - python connector for Amazon Web Services.
* pandas - a python data analysis library

## Configuration

The program has two configuration files:

* keys.json - amazon access keys
* bot.json - chatbot configuration

## Usage

The main program lies in the `lex_tester.py` file. It reads excel files in the input folder and reads input text from the column `Sample Utterances`. It then uses the amazon api to get a response from the chatbot and writes it to a file in the output folder.

## Known issues

This program failed to test the chatbot it was originally built for due to the following error:

```
DependencyFailedException: An error occurred (DependencyFailedException) when calling the PostText operation: Invalid Lambda Response: Received error response from Lambda (LambdaRequestId: af4e1770-7987-4e7c-92fd-063369d9d0e6; Error: Unhandled)
```

As far as my research went, this is due to missing parameters in the lambda function triggered by the intents. In particular these lines:

```
dialogAction: {
  type: "ElicitIntent",
  message: {
    contentType: "CustomPayload",
    content: reply
  }
```

What is missing is the `fulfillmentState` parameter, which can be set as follows:

```
dialogAction: {
  type: "ElicitIntent",
  fulfillmentState: "Fulfilled",
  message: {
    contentType: "CustomPayload",
    content: reply
  }
```

More on amazon official documents: https://docs.aws.amazon.com/lex/latest/dg/lambda-input-response-format.html
