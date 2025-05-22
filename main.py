"""
Simple tool to check how long it has been since your last block proposal and how likely this scenario is.

Note: one-time key registration is considered and it is assumed the evaluated account is currently staking.

Disclaimer: the code serves to illustrate an example and provides no guarantees of correctness.
"""
import json
import argparse
import requests
from typing import Tuple
from datetime import timedelta

import numpy as np
from algosdk.v2client.algod import AlgodClient
from algosdk.v2client.indexer import IndexerClient

from utils import (
    get_round_of_latest_block_reward, 
    convert_round_delta_to_time_delta,
    get_block_rewards
)


def get_online_stake() -> int:
    """Get the current online stake from Nodely.

    Notes
    -----
    https://afmetrics.api.nodely.io/v1/api-docs/#get-/v1/realtime/participation/online

    Returns
    -------
    int
        Online stake.
    """
    url = "https://afmetrics.api.nodely.io/v1/realtime/participation/online"
    response = requests.get(url)
    return int(response.json()['stake_micro_algo'])


def get_online_stake_history(
    round_start: int,
    round_end: int
) -> dict:
    """Get the online stake history within a certain time window from Nodely.

    Notes
    -----
    Online stake is averaged across 10 rounds.
    https://afmetrics.api.nodely.io/v1/api-docs/#get-/v1/delayed/totalstake/history

    Parameters
    ----------
    round_start: int
        First round in observation window.
    round_end: int
        End round in observation window.

    Returns
    -------
    dict
        Online stake per 10 rounds in the observation window.
    """
    url = "https://afmetrics.api.nodely.io/v1/delayed/totalstake/history"
    params = {
        "from": f"{round_start}",
        "to": f"{round_end}"
    }
    headers = {"accept": "*/*"}
    response = requests.get(url, params=params, headers=headers)
    data = [json.loads(line) for line in response.text.strip().split('\n')]
    total_stake = np.array([d['total_stake'] for d in data])
    total_stake = total_stake.round()
    total_stake = [int(ts) for ts in total_stake]
    round_list = [f'{r}' for r in range(round_end, round_start, -10)]
    stake_per_round = dict(zip(round_list, total_stake))
    return stake_per_round


def get_account_balance_history(
    indexer_client: IndexerClient,
    address: str,
    round_start: int,
    round_end: int,
    limit: int=1000
) -> dict:
    """Get account balance within a certain time window from Nodely.

    Notes
    -----
    https://mainnet-idx.4160.nodely.dev/x2/api-docs/#tag/accounts

    Parameters
    ----------
    indexer_client: IndexerClient
        Indexer client.
    address: str
        Algorand account address.
    round_start: int
        first round in observation window.
    round_end: int
        Last round in observation window.
    limit: int, optional
        Max. number of fetched transaction. Default is 1000.

    Returns
    -------
    dict
        Online stake for each round that the balance changed within the observation window.
    """
    # account_balance = np.array([], dtype=int)
    # for round_num in range(round_end, round_start, -10):
    #     url = f"https://mainnet-idx.4160.nodely.dev/x2/account/{address}/snapshot/{round_num}/0"
    #     headers = {"accept": "*/*"}
    #     response = requests.get(url, headers=headers)
    #     balance_on_round = int(response.json()['balance']*10**-6)
    #     account_balance = np.r_[account_balance, balance_on_round]
    # Get final balance
    url = f"https://mainnet-idx.4160.nodely.dev/x2/account/{address}/snapshot/{round_end}/0"
    headers = {"accept": "*/*"}
    response = requests.get(url, headers=headers)
    balance_previous = response.json()['balance']
    # Loop over transactions and derive balance on round of transaction
    response = indexer_client.search_transactions_by_address(address, limit=limit)
    transactions = response['transactions']
    account_balance = dict()
    for idx, txn in enumerate(transactions):
        account_balance[f'{round_end}'] = int(round(balance_previous*10**-6))
        if txn['confirmed-round'] >= round_end:
            continue
        if txn['confirmed-round'] < round_start:
            break
        if txn['tx-type']=='pay':
            txn_amount = txn['payment-transaction']['amount']
            if txn['payment-transaction']['receiver'] == address:
                balance_previous += txn_amount
            else:
                balance_previous -= txn_amount
    return account_balance


def calculate_likelihood_of_no_rewards_simple(
    account_stake: int, 
    total_online_stake: int,
    rounds_since_last_reward: int
) -> float:
    """Calculate the likelihood based on the current amount.

    Notes
    -----
    Uses only the current account and total online stake.
    A more correct version would consider past values, per round or in batches of rounds (time window).

    Parameters
    ----------
    account_stake : int
        Amount of stake that the account owns.
    total_online_stake : int
        Amount of stake that is currently participating in consensus.
    rounds_since_last_reward : int
        The number of elapsed rounds since the last reward.

    Returns
    -------
    float
        Likelihood of no rewards happening.
    """
    percentage_of_total_stake = account_stake / total_online_stake
    return (1 - percentage_of_total_stake)**rounds_since_last_reward


def calculate_likelihood_of_no_rewards(
    account_stake: dict, 
    total_stake: dict
) -> float:
    """Calculate the likelihood based on the account and total stake history.

    Parameters
    ----------
    account_stake : dict
        Amount of stake that the account owned during the time window.
    total_stake : dict
        Amount of stake during the desired time window.

    Returns
    -------
    float
        Likelihood of no rewards happening.
    """
    account_stake_keys = np.array(list(account_stake.keys())).astype(int)[::-1]
    account_stake_values = np.array(list(account_stake.values())).astype(int)[::-1]
    total_stake_keys = np.array(list(total_stake.keys())).astype(int)
    total_stake_values = np.array(list(total_stake.values())).astype(int)
    trunc_total_stake_keys = np.copy(total_stake_keys)
    portion_of_total_stake = np.array([])
    weights = np.array([])
    for ask, asv in zip(account_stake_keys, account_stake_values): # Iterate from lower round number to higher round number
        mask = trunc_total_stake_keys <= ask
        mean_total_stake = np.mean(total_stake_values[mask])
        portion_of_total_stake = np.r_[portion_of_total_stake, asv / mean_total_stake]
        trunc_total_stake_keys = trunc_total_stake_keys[np.logical_not(mask)] # Prep for next iteration
        weights = np.r_[weights, np.sum(mask) / total_stake_keys.size]
    mean_portion_of_total_stake = np.mean(portion_of_total_stake*weights)
    number_of_rounds = int(np.diff(total_stake_keys[[-1, 0]])[0])
    return (1 - mean_portion_of_total_stake)**number_of_rounds


def get_no_rewards_stats(
    algod_client: AlgodClient,
    indexer_client: IndexerClient,
    address: str
) -> Tuple[float, float]:
    """Get the stats for not receiving any rewards.

    Parameters
    ----------
    algod_client : AlgodClient
        Algod client.
    indexer_client : IndexerClient
        Indexer client.
    address : str
        Address of the targeted account.

    Returns
    -------
    Tuple[float, float]
        The amount of seconds since the last produced block and the likelihood of going with no rewards for this long.
    """
    current_round = algod_client.status()['last-round']
    round_of_latest_block_reward = get_round_of_latest_block_reward(
        indexer_client=indexer_client,
        address=address,
        first_round=current_round-1_000_000, # Roughly one month
        last_round=current_round
    )
    rounds_since_last_reward = current_round - round_of_latest_block_reward
    total_online_stake = get_online_stake_history(
        round_start=round_of_latest_block_reward,
        round_end=current_round
    )
    account_stake = get_account_balance_history(
        indexer_client=indexer_client,
        address=address,
        round_start=round_of_latest_block_reward,
        round_end=current_round
    )
    likelihood_of_no_rewards = calculate_likelihood_of_no_rewards(
        account_stake,
        total_online_stake
    )
    return (
        convert_round_delta_to_time_delta(rounds_since_last_reward), 
        likelihood_of_no_rewards
    )


def get_anticipated_proposal_period_simple(
    indexer_client: IndexerClient,
    address: str
) -> float:
    """Get the anticipated block proposal period.

    Notes
    -----
    Uses only the current account and total online stake.
    A more correct version would consider past values, per round or in batches of rounds (time window).

    Parameters
    ----------
    indexer_client : IndexerClient
        Indexer client.
    address : str
        Address of the targeted account.

    Returns
    -------
    float
    """
    total_online_stake = get_online_stake()
    account_stake = indexer_client.account_info(
        address=address
    )['account']['amount']
    percentage_of_stake = account_stake / total_online_stake
    round_period_s = 3
    return round_period_s / percentage_of_stake / 3600


def get_average_proposal_period(
    algod_client: AlgodClient,
    indexer_client: IndexerClient,
    address: str
) -> Tuple[int, float]:
    """Get the average block proposal period.

    Parameters
    ----------
    algod_client : AlgodClient
        Algod client.
    indexer_client : IndexerClient
        Indexer client.
    address : str
        Address of the targeted account.

    Returns
    -------
    Tuple[int, float]
    """
    _, tx_timestamp, _, _ = get_block_rewards(
        algod_client,
        indexer_client,
        address
    )
    return tx_timestamp.size, int(np.mean(np.diff(tx_timestamp))) / 3600


def main(address):
    algod_client = AlgodClient(
        '', 
        'https://mainnet-api.4160.nodely.dev'
    )
    indexer_client = IndexerClient(
        '',  
        'https://mainnet-idx.algonode.cloud'
    )

    # Try fetching address and trigger error if non-existent
    indexer_client.account_info(address)
    print(f'\nLooking into address {address}')

    anticipated_proposal_period_h = get_anticipated_proposal_period_simple(
        indexer_client=indexer_client,
        address=address
    )
    population_size, average_proposal_period_h = get_average_proposal_period(
        algod_client=algod_client,
        indexer_client=indexer_client,
        address=address
    )
    print(f'*The anticipated block production time is about: {round(anticipated_proposal_period_h, 1)} h')
    print(f'*The number of blocks produced so far is: {population_size}')
    print(f'*The average block production time is about: {round(average_proposal_period_h, 1)} h')

    seconds_without_reward, likelihood_of_no_rewards = get_no_rewards_stats(
        algod_client=algod_client,
        indexer_client=indexer_client,
        address=address
    )
    likelihood_of_no_rewards_perc = round(likelihood_of_no_rewards*100, 2)
    print('*Duration since last produced block: ' + str(timedelta(seconds=seconds_without_reward)))
    print(f'*Likelihood of not producing a block during this time: {likelihood_of_no_rewards_perc} %')
    print()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check how long it has been since your last produced block and how likely this scenario is."
    )
    parser.add_argument("address", help="Address of Algorand account.")
    address = parser.parse_args().address
    main(address)
