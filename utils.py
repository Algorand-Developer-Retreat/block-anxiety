from typing import Tuple

import numpy as np
from algosdk.v2client.algod import AlgodClient
from algosdk.v2client.indexer import IndexerClient


REWARD_ADDRESS = 'Y76M3MSY6DKBRHBL7C3NNDXGS5IIMQVQVUAB6MP4XEMMGVF2QWNPL226CA'

FIRST_ROUND_WITH_REWARDS = 46525262 # Approximate (slightly after upgrade)


def convert_round_delta_to_time_delta(
    round_delta: int,
    round_duration_s: int | float = 3
) -> float:
    """Convert from duration in rounds to duration in seconds.

    Parameters
    ----------
    round_delta : int
        Difference in rounds.
    round_duration_s : int | float, optional
        Duration of a single round (assumed constant or average), by default 3.

    Returns
    -------
    float
        Difference in seconds.
    """
    return round_delta * round_duration_s


def get_transaction_of_latest_block_reward(
    indexer_client: IndexerClient,
    address: str,
    first_round: int,
    last_round: int
) -> dict:
    """Get the transaction associated with the latest block reward.

    Parameters
    ----------
    indexer_client : IndexerClient
        Indexer client.
    address : str
        Address of the tracked account.
    first_round : int
        Round starting with which the transactions are checked for a block reward.
    last_round : int
        Round ending with which the transactions are checked for a block reward.

    Returns
    -------
    dict
        Latest block reward transaction.
    """
    response = indexer_client.search_transactions_by_address(
        address=address,
        min_round=first_round,
        max_round=last_round
    )
    for tx in response['transactions']:
        if tx['tx-type']=='pay':
            if tx['sender'] == REWARD_ADDRESS:
                return tx

def get_round_of_latest_block_reward(
    indexer_client: IndexerClient,
    address: str,
    first_round: int,
    last_round: int
) -> int:
    """Get the round at which the latest block reward transaction was confirmed.

    Parameters
    ----------
    indexer_client : IndexerClient
        Indexer client.
    address : str
        Address of the tracked account.
    first_round : int
        Round starting with which the transactions are checked for a block reward.
    last_round : int
        Round ending with which the transactions are checked for a block reward.

    Returns
    -------
    int
        Latest block reward round.
    """
    return get_transaction_of_latest_block_reward(
        indexer_client =indexer_client,
        address=address,
        first_round=first_round,
        last_round=last_round
    )['confirmed-round']


def get_block_rewards(
    algod_client: AlgodClient,
    indexer_client: IndexerClient,
    address: str
) -> Tuple[list, list, list]:
    """Get all the block rewards that an account has received.

    Parameters
    ----------
    algod_client : AlgodClient
        Algod client.
    indexer_client : IndexerClient
        Indexer client.
    address : str
        Algorand account address.

    Returns
    -------
    Tuple[list, list, list]
    """
    return get_transaction_list(
        indexer_client,
        address,
        first_round=FIRST_ROUND_WITH_REWARDS,
        last_round=algod_client.status()['last-round']
    )


def get_transaction_list(
    indexer_client: IndexerClient,
    address: str,
    first_round: int,
    last_round: int
) -> Tuple[list, list, list]:
    """Get all the transactions of an account.

    Parameters
    ----------
    algod_client : AlgodClient
        Algod client.
    indexer_client : IndexerClient
        Indexer client.
    address : str
        Algorand account address.
    first_round: int
        First round for checking.
    last_round: int
        Last round for checking.

    Returns
    -------
    Tuple[list, list, list]
    """
    response = indexer_client.search_transactions_by_address(
        address=address,
        min_round=first_round,
        max_round=last_round
    )
    amount = np.array([], dtype=int)
    timestamp = np.array([], dtype=int)
    block = np.array([], dtype=int)
    sender = []
    for tx in response['transactions']:
        if tx['tx-type']=='pay':
            amount = np.r_[amount, int(tx['payment-transaction']['amount'])]
            if tx['sender'] == address:
                amount[-1] *= -1 # Make outbound payments a negative change
            timestamp = np.r_[timestamp, int(tx['round-time'])]
            confirmed_round = np.r_[block, int(tx['confirmed-round'])]
            sender.append(tx['sender'])
    # Rotate to have oldest first
    return amount[::-1], timestamp[::-1], confirmed_round[::-1], np.array(sender)[::-1]
