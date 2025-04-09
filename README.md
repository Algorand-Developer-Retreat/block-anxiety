# Block Anxiety

A tool to assess the time since your last produced block on the Algorand blockchain and estimate how probable this situation is, based on your stake. The main features are:
- Calculate **time since last produced block**
- Estimate **likelihood of not producing a block**
- Derive **anticipated** and **average** block production time

The motivation for implementing this tool and high-level information on the probability of (not) producing a block can be found in [Valar's blog post "Where is my block?"](https://valar-staking.medium.com/where-is-my-block-70113da4d817)

> Disclaimer: This tool is for illustrative purposes only. No guarantees of correctness are provided. 


## Usage

### Prerequisites

Install required packages with:

```bash
python3 -m pip install -r requirements.txt
```

### Running

Run the evaluation using:

```bash
python3 main.py <ALGOD_ACCOUNT_ADDRESS>
```

### Output

You should see the following data:

- Anticipated block production period (hours)
- Total number of produced blocks
- Average block production period (hours)
- Duration since last produced block
- Likelihood (in %) of not producing a block for that duration


## Notes

The tool assumes that the provided account is staking and has completed one-time key registration. 
Moreover, it assumes the only transactions associated with the account while staking were due to received block rewards.
