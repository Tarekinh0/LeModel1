**LeModel1**

If you have any questions, please contact me @ : tarek.mahfoudh@tartech.net
Or simply raise an issue on GitHub to request a change or signal a bug.
Have fun :)

We will need to reconfigure the libraries before making the main.py work.
Here are the dependencies that you will need to make sure everything works locally without any API calls to HuggingFace.

# Detoxify 
Don't install detoxify from HuggingFace, don't install it from Github, install it from this repo. Just launch the command:
```
pip install -e detoxify
```
# TheBloke/deepseek-coder-1.3b-instruct-GPTQ

```
git lfs clone https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GPTQ
```
No weird modifications to do. Works out of the box.

# Bark-Small
```
git lfs clone https://huggingface.co/suno/bark-small-GPTQ
```
We also use the transformers library to interact with it. Nothing weird to do, works out of the box.
# Requirements.txt
Now that we have all the libraries in place just install the requirements.txt
```
pip install requirements.txt
```
Launch the program. The main.py should work :)




## What bugs did we fix in Detoxify ?
Detoxify only works with the version 4.30.0 of the Transformes library (HuggingFace) and all the others (and our code) work with the 4.35.0
We had to modify the following to have it updated.
This is a description of the modifications:

Get detoxify from github (not from HuggingFace) 
```
git lfs clone https://github.com/unitaryai/detoxify
```

go and modify detoxify/detoxify.py et and change the function get_model_and_tokenizer by this one (we added to all the 'from_pretrained' the boolean 'local_files_only = True' + we save them (model & tokenizer) in a local file with the method '.save_pretrained(PATH)' to reuse them instead of calling the API):
```
def get_model_and_tokenizer(model_type, model_name, tokenizer_name, num_classes, state_dict, huggingface_config_path=None):

	model_class = getattr(transformers, model_name)

	model = model_class.from_pretrained(

		pretrained_model_name_or_path=None,

		config=huggingface_config_path or model_type,

		num_labels=num_classes,

		state_dict=state_dict,

		# local_files_only=huggingface_config_path is not None,

		local_files_only=True,

	)

	tokenizer_class = getattr(transformers, tokenizer_name)

	tokenizer = tokenizer_class.from_pretrained(

		huggingface_config_path or model_type,

		# local_files_only=huggingface_config_path is not None,

		local_files_only=True,

		# TODO: may be needed to let it work with Kaggle competition

		model_max_length=512,

	)

	model.save_pretrained("./libs/saved/detoxpre")

	tokenizer.save_pretrained("./libs/saved/detoxpre")

return model, tokenizer
```

You also need to change the setup.py and the requirements.txt by changing the lines "transformers == 4.30.0" to "transformers >= 4.30.0"
