# rasa actions

A readme on how to create and use custom action files such as `actions.py` in the `actions` folder.

## responses

```
dispatcher.utter_message(
    template="utter_greet",
    name="Sara"
)
```

When using a custom action server:

```
{
  "events":[
    ...
  ],
  "responses":[
    {
      "template":"utter_greet",
      "name":"Sara"
    }
  ]
}
```
