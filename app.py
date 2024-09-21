import gradio as gr
import openai
import json
import time

def make_api_call(client, model, messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(api_key, api_base, model, prompt):
    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES."""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(client, model, messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data['next_action'] == 'final_answer' or step_count > 25:
            break
        
        step_count += 1

    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
    
    start_time = time.time()
    final_data = make_api_call(client, model, messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data['content'], thinking_time))

    return format_steps(steps, total_thinking_time)

def format_steps(steps, total_time):
    html_content = ""
    for title, content, thinking_time in steps:
        if title == "Final Answer":
            html_content += f"<h3>{title}</h3>"
            html_content += f"<p>{content}</p>"
        else:
            html_content += f"""
            <details>
                <summary><strong>{title}</strong></summary>
                <p>{content}</p>
                <p><em>Thinking time for this step: {thinking_time:.2f} seconds</em></p>
            </details>
            <br>
            """
    html_content += f"<strong>Total thinking time: {total_time:.2f} seconds</strong>"
    return html_content

def main(api_key, api_base, model, user_query):
    if not api_key or not api_base:
        return "Please enter your OpenAI API key and base URL to proceed."
    
    if not model:
        return "Please enter a model name to proceed."
    
    if not user_query:
        return "Please enter a query to get started."
    
    try:
        return generate_response(api_key, api_base, model, user_query)
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define the Gradio interface
iface = gr.Interface(
    fn=main,
    inputs=[
        gr.Textbox(label="OpenAI API Key", type="password"),
        gr.Textbox(label="OpenAI API Base URL", value="https://api.openai.com/v1"),
        gr.Textbox(label="Model Name", value="gpt-3.5-turbo"),
        gr.Textbox(label="Enter your query", lines=2)
    ],
    outputs="html",
    title="g1: Using OpenAI to create o1-like reasoning chains",
    description="This is an early prototype of using prompting to create o1-like reasoning chains to improve output accuracy."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
