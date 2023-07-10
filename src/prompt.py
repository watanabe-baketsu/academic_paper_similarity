import json


system_inputs = "```" \
                "1.\ntitle : paper title\nabstract：paper abstract\n" \
                "2.\ntitle : paper title\nabstract : paper abstract\n\n" \
                "```"
output_format = [
    {
        "論文のタイトル": "title",
        "概要": "summarize the paper simply",
        "長所": "pros",
        "短所": "cons"
    },
    {
        "論文のタイトル":  "title",
        "概要": "summarize the paper simply",
        "長所": "pros",
        "短所": "cons"
    },
]
prompt_system = "You are an assistant tasked with summarizing information from academic papers. " \
                "Please compile the information from the given five papers and output it in JSON format. " \
                "You should output the JSON in Japanese. " \
                f"The information for the papers will be provided in the following format:\n{system_inputs}\n" \
                f"Please output the JSON in the following format:\n{json.dumps(output_format)}"

