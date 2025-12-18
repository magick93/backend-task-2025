
## Simplified Task Description

Here's the task broken down simply:

### What You're Building
A serverless API that groups similar sentences into meaningful clusters and analyzes them.

### The Flow
1. **Receive** JSON with sentences (all about the same theme, e.g., "account")
2. **Cluster** similar sentences into sub-groups (e.g., "login problems", "password issues")
3. **Analyze** each cluster for sentiment (positive/negative/neutral)
4. **Generate** 2-3 key insights per cluster
5. **Return** JSON with organized clusters

### Core Components Needed
1. **API endpoint** (AWS API Gateway + Lambda)
2. **Clustering logic** (group similar sentences)
3. **Sentiment analysis** (classify each cluster)
4. **Insight generation** (summarize what each cluster is about)

### Key Points
- Sentences have IDs (from comments)
- Same ID can appear multiple times in input (different sentences, same comment)
- Same ID should only appear once per cluster in output
- Focus on infrastructure/architecture over perfect AI results
- 4-hour time limit - show your approach, not a perfect solution

### The "Extra" Part
There's also a comparison mode (baseline vs comparison datasets), but that's secondary - design with it in mind but implement the basic version first.


# The task

We have a basic function working in `/home/anton/git/backend-task-2025/text-analysis-service` that can be run locally using `sam local invoke TextAnalysisFunction --event test_event.json`.

Review `READ


Update the structure, tests and functions in `text-analysis-service` folder to match `The Flow` described above. 
Your approach should be to keep running the unit tests and ensure that as you migrate to the AWS sam / serverless approach, the unit tests continue to run and pass. An example of the amazon web services serverless structure and syntax can be found in `aws-sam-sample/text-analysis`.

You should also test that the functions work locally using `sam local invoke <function_logical_id> [options]`





