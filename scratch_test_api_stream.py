import requests
import json

def test_api_stream():
    url = "http://localhost:8000/v1/query"
    payload = {
        "query": "Summarise the Q3 earnings report and identify risks",
        "stream": True
    }
    
    print(f"Submitting query to {url}...")
    response = requests.post(url, json=payload, stream=True)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    print("Stream opened. Waiting for events...")
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("event:"):
                event_type = decoded_line.split("event: ")[1]
                print(f"\n[Event: {event_type}]")
            elif decoded_line.startswith("data:"):
                data = json.loads(decoded_line.split("data: ")[1])
                if event_type == "done":
                    print("FINAL ANSWER RECEIVED:")
                    print(json.dumps(data, indent=2))
                else:
                    agent = data.get("data", {}).get("agent", "unknown")
                    type_ = data.get("data", {}).get("type", "unknown")
                    print(f"  Agent: {agent} | Type: {type_}")
                    # Capture run_id for trace fetching
                    run_id = data.get("run_id")

    if run_id:
        print(f"\n" + "-"*40)
        print(f"FETCHING EXECUTION TRACE: {run_id}")
        print("-"*40)
        trace_url = f"http://localhost:8000/v1/runs/{run_id}/trace"
        trace_resp = requests.get(trace_url)
        if trace_resp.status_code == 200:
            trace_data = trace_resp.json()
            print(f"Status: {trace_data['status']}")
            print(f"Total Latency: {trace_data['total_latency_ms']}ms")
            print(f"Total Tokens: {trace_data['total_tokens']}")
            print("\nSteps Audit Trail:")
            for step in trace_data['steps']:
                print(f"  [{step['step_index']}] {step['agent_name']} ({step['latency_ms']}ms)")
                print(f"      Out: {step['output_summary'][:150]}...")
        else:
            print(f"Failed to fetch trace: {trace_resp.status_code}")

if __name__ == "__main__":
    test_api_stream()
