"""
Simple tool to check how long it has been since your last block proposal and how likely this scenario is.

Note: one-time key registration is considered and it is assumed the evaluated account is currently staking.

Disclaimer: the code serves to illustrate an example and provides no guarantees of correctness.
"""
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

    Returns
    -------
    int
        Online stake.
    """
    url = "https://afmetrics.api.nodely.io/v1/realtime/participation/online"
    response = requests.get(url)
    response.json()
    return int(response.json()['stake_micro_algo'])


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
    total_online_stake = get_online_stake()
    account_stake = indexer_client.account_info(
        address=address
    )['account']['amount']
    current_round = algod_client.status()['last-round']
    round_of_latest_block_reward = get_round_of_latest_block_reward(
        indexer_client=indexer_client,
        address=address,
        first_round=current_round-1_000_000, # Roughly one month
        last_round=current_round
    )
    rounds_since_last_reward = current_round - round_of_latest_block_reward
    likelihood_of_no_rewards = calculate_likelihood_of_no_rewards_simple(
        account_stake,
        total_online_stake, 
        rounds_since_last_reward
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
