"""
Subject Code: FNCE30010
Subject Name: Algorithmic Trading
Full Name: Joel Thomas
Assignment Name: Project 1 - Task 2 code (CAPM)
"""

from fmclient import Agent
from fmclient import Order, OrderSide, OrderType
import numpy as np
import copy

# Submission details
SUBMISSION = {"number": "915951", "name": "Joel Thomas"}

# Defining global variables
PROFIT_MARGIN = 10                              # Stores the minimum amount of required profit (in cents) per trade
DOLLAR_TO_CENTS = 100                           # Converts dollars to cents (multiply) and cents to dollars (divide)
MINIMUM_SETTLED_CASH = 5*DOLLAR_TO_CENTS        # Minimum required cash (cents) for trades, need to sell notes otherwise
BUY_UNITS_INDEX = 0                             # Index in orders list to denote buy units
SELL_UNITS_INDEX = 1                            # Index in orders list to denote sell units
BUY_PRICE_INDEX = 2                             # Index in orders list to denote buy price
SELL_PRICE_INDEX = 3                            # Index in orders list to denote sell price


class CAPMBot(Agent):
    """
    Trading bot that takes one side only, reactive, and trades in multiple markets by buying if price of security is
    below fair value and selling otherwise. Fair value based on expected liquidating dividend amount for each security.
    Aims to trade only if the current performance of the proposed new holdings is greater than the previous performance
    of the market holdings. Sells notes if little settled cash at any point during trading session. Avoids short
    selling.
    """
    def __init__(self, account, email, password, marketplace_id, risk_penalty=0.01, session_time=20):
        """
        Constructor for the Bot.
        :param account: account name
        :param email: email id
        :param password: password
        :param marketplace_id: id of the marketplace
        :param risk_penalty: penalty for risk
        :param session_time: total trading time for one session
        """
        super().__init__(account, email, password, marketplace_id, name="CAPM Bot")
        self.description = "Algorithmic trading bot based on CAPM"
        self._payoff_matrix = []                            # Size of liquidating dividend in each state for all assets
        self._risk_penalty = risk_penalty                   # Penalty for risk (performance evaluation)
        self._session_time = session_time                   # Current session time
        self._market_ids = {}                               # Market ids for all assets
        self._market_properties = {}                        # Stores minimum, maximum and tick size for all securities
        self._settled_cash = -1                             # Current settled cash balance
        self._available_cash = -1                           # Current available cash balance
        self._market_holdings = np.array([])                # Stores available units for each asset
        self._proposed_holdings = np.array([])              # Stores proposed available units for each asset
        self._proposed_cash = -1                            # Final settled cash if trade proposed holdings
        self._expected_state_payoffs = np.array([])         # Expected state payoffs for each row in payoff matrix
        self._covariance_matrix = None                      # Covariance matrix for payoffs
        self._previous_performance = None                   # Performance based on held cash and market holdings
        self._current_performance = None                    # Performance based on proposed cash and proposed holdings
        self._waiting_for_servers = {}                      # Waiting for server flag for each security
        self._counter = 0                                   # Counts number of times a function executes
        self._num_executed_trades = {}                      # Number of trades executed for each security
        self._no_initial_signals = []                       # When bot initialised, signals found through first optimal
                                                            # order, hence no initial signal (buy or sell) for any
                                                            # security. Set to True for each asset during initialisation
        self._signals = []                                  # To prevent market cornering, next trade on same security
                                                            # must be on opposite side. True for buy, False for sell
        self._optimal_order = []                            # Contains optimal order consisting of buy units, sell
                                                            # units, buy price and sell price for a security
        self._last_order_ref = None                         # Last order reference

    def initialised(self):
        """
        Stores market properties, initialises variables and calculates payoff and covariance matrices.
        :return:
        """
        # State (column) payoffs for each security
        state_payoffs = []
        for market_id, market_info in self.markets.items():
            security = market_info["item"]
            # Storing market id, minimum, maximum and tick size for each security
            self._market_ids[security] = market_info["id"]
            self.inform(f"{security} market id: {self._market_ids[security]}")
            self._market_properties[security] = {}
            self._market_properties[security]["Min price"] = market_info["minimum"]
            self._market_properties[security]["Max price"] = market_info["maximum"]
            self._market_properties[security]["Tick"] = market_info["tick"]
            description = market_info["description"]
            # Splits string and converts to integer to separate payoffs
            payoffs = [int(dividend) for dividend in description.split(",")]
            state_payoffs.append(payoffs)
            # Expected state payoffs (fair value) for each security in dollar form
            self._expected_state_payoffs = np.append(self._expected_state_payoffs, np.mean(payoffs)/DOLLAR_TO_CENTS)
            self._waiting_for_servers[security] = False
            self._num_executed_trades[security] = 0
            # Initial buy or sell depends on prices in order book for security, hence no initial signal for each asset
            self._no_initial_signals.append(True)
            self._signals.append(None)
        # Payoff matrix in dollar form
        self._payoff_matrix = np.array(state_payoffs)/DOLLAR_TO_CENTS
        self._covariance_matrix = np.cov(self._payoff_matrix, ddof=0)

    def _best_bid_and_ask(self, order_book, security):
        """
        Finds best bid and ask prices in the order book of a security.
        :param order_book: current order book of security
        :param security: security identified from call to received_order_book
        :return:
        """
        # Start with worst bid = minimum market price and worst ask = highest market price
        best_bid = self._market_properties[security]["Min price"]
        best_ask = self._market_properties[security]["Max price"]
        # Find best bids and asks
        for o in order_book:
            if not o.mine and o.side == OrderSide.BUY and o.price > best_bid:
                best_bid = o.price
            elif not o.mine and o.side == OrderSide.SELL and o.price < best_ask:
                best_ask = o.price
        return best_bid, best_ask

    def _force_sell_notes(self, best_bid, best_ask, expected_payoff, security, correct_index):
        """
        If insufficient settled cash according to pre-determined threshold, attempts to force sell settled notes to
        generate cash.
        :param best_bid: best bid price in order book
        :param best_ask: best ask price in order book
        :param expected_payoff: fair value of security based on liquidating dividend at end of session
        :param security: security identified from call to received_order_book
        :param correct_index: correct index of security based on layout identified during initialisation
        :return: no_change_order: order list such that performance does not change so that _execute_strategy is not
        called again under received_order_book
        """
        if best_bid > expected_payoff*DOLLAR_TO_CENTS - PROFIT_MARGIN and not self._waiting_for_servers[security] and \
                self._market_holdings[correct_index] > 0:
            # Force sell to generate cash
            buy_units = 0
            sell_units = 1
            self._optimal_order = [buy_units, sell_units, best_ask, best_bid]
            self._execute_strategy(self._optimal_order, security)
            # Because order is executed here, do not want to check performance and then execute again under
            # received_order_book and _execute_strategy. No change order will ensure current performance = previous
            # performance
            sell_units = 0
            no_change_order = [buy_units, sell_units, best_ask, best_bid]
            return no_change_order

    def _optimal_order_creation(self, order_book, security):
        """
        Generates the optimal order list that should increase performance if executed.
        :param order_book: current order book of security
        :param security: security identified from call to received_order_book
        :return:
        """
        best_bid, best_ask = self._best_bid_and_ask(order_book, security)
        # Find correct index of security based on layout identified during initialisation to access expected payoff
        correct_index = list(self._market_ids.keys()).index(security)
        expected_payoff = self._expected_state_payoffs[correct_index]
        # Need to sell notes at no lower than a pre-determined price (can be a loss) to generate sufficient cash until
        # threshold is reached
        if security == "note" and self._settled_cash*DOLLAR_TO_CENTS < MINIMUM_SETTLED_CASH:
            return self._force_sell_notes(best_bid, best_ask, expected_payoff, security, correct_index)
        # First time trading a security, optimal order based only on prices and not on any signals
        if self._no_initial_signals[correct_index]:
            # Buy at best_ask if profitable
            if best_ask < expected_payoff*DOLLAR_TO_CENTS - PROFIT_MARGIN:
                buy_units = 1
                self._signals[correct_index] = False
            else:
                buy_units = 0
            # Sell at best bid if profitable
            if best_bid > expected_payoff*DOLLAR_TO_CENTS + PROFIT_MARGIN:
                sell_units = 1
                self._signals[correct_index] = True
            else:
                sell_units = 0
            self._no_initial_signals[correct_index] = False
        # Not first time trading a security, use signals to prevent cornering market or only have buy or sell orders
        else:
            # Buy at best ask if profitable and last order was sell order
            if best_ask < expected_payoff*DOLLAR_TO_CENTS - PROFIT_MARGIN and self._signals[correct_index]:
                buy_units = 1
                self._signals[correct_index] = False
            else:
                buy_units = 0
            # Sell at best bid if profitable and last order was buy order
            if best_bid > expected_payoff*DOLLAR_TO_CENTS + PROFIT_MARGIN and not self._signals[correct_index]:
                sell_units = 1
                self._signals[correct_index] = True
            else:
                sell_units = 0
        self._optimal_order = [buy_units, sell_units, best_ask, best_bid]
        return self._optimal_order

    # orders should be [buy units, sell units, buy price, sell price] and not for every security
    def get_potential_performance(self, orders, security):
        """
        Returns the portfolio performance if the given list of orders is executed.
        The performance as per the following formula:
        Performance = ExpectedPayoff - b * PayoffVariance, where b is the penalty for risk.
        :param security: security identified from call to received_order_book
        :param orders: list of orders
        :return:
        """
        buy_units = orders[BUY_UNITS_INDEX]
        sell_units = orders[SELL_UNITS_INDEX]
        buy_price = orders[BUY_PRICE_INDEX]
        sell_price = orders[SELL_PRICE_INDEX]
        # Finding change in cash held given list of orders
        total_buy_price = buy_units * buy_price
        total_sell_price = sell_units * sell_price
        total_cash_change = total_sell_price - total_buy_price
        # Proposed cash in dollar form
        self._proposed_cash = self._settled_cash + total_cash_change/DOLLAR_TO_CENTS
        # Find correct index of security based on layout identified during initialisation to access expected payoff
        correct_index = list(self._market_ids.keys()).index(security)
        # Finding change in holdings given list of orders
        self._proposed_holdings = np.array([])
        for i in range(len(self._market_ids)):
            # Orders based on one security only so proposed holdings should change only for relevant security
            if i == correct_index:
                self._proposed_holdings = np.append(self._proposed_holdings,
                                                    self._market_holdings[i] + orders[BUY_UNITS_INDEX] -
                                                    orders[SELL_UNITS_INDEX])
            # Holdings of other securities remain the same
            else:
                self._proposed_holdings = np.append(self._proposed_holdings, self._market_holdings[i])
        # Matrix calculations of expected payoff and payoff variance
        previous_expected_payoff = self._settled_cash + np.dot(self._market_holdings, self._expected_state_payoffs)
        previous_payoff_variance = np.dot(self._market_holdings, np.dot(self._covariance_matrix, self._market_holdings))
        current_expected_payoff = self._proposed_cash + np.dot(self._proposed_holdings, self._expected_state_payoffs)
        current_payoff_variance = np.dot(self._proposed_holdings, np.dot(self._covariance_matrix,
                                                                         self._proposed_holdings))
        # Performance formula
        self._previous_performance = previous_expected_payoff - self._risk_penalty * previous_payoff_variance
        self._current_performance = current_expected_payoff - self._risk_penalty * current_payoff_variance
        # Periodically outputs current performance once counter reaches 10
        if self._counter == 20:
            self.inform("The current performance is: {0}".format(self._current_performance))
            # Reset counter to 0 once 10 is reached
            self._counter = 0
        self._counter += 1

    def is_portfolio_optimal(self, order_book, security):
        """
        Returns true if the current holdings are optimal (as per the performance formula), false otherwise.
        :return: self._current_performance <= self._previous_performance: if false, optimal to execute optimal orders
        """
        self.get_potential_performance(self._optimal_order_creation(order_book, security), security)
        return self._current_performance <= self._previous_performance

    def _execute_strategy(self, optimal_order, security):
        """
        Attempts to execute the optimal order list to increase performance or informs the user if unable to
        :param optimal_order: optimal order list for a security generated by _optimal_order_creation and best bid and
        ask prices
        :param security: security identified from call to received_order_book
        :return:
        """
        buy_units = optimal_order[BUY_UNITS_INDEX]
        sell_units = optimal_order[SELL_UNITS_INDEX]
        buy_price = optimal_order[BUY_PRICE_INDEX]
        sell_price = optimal_order[SELL_PRICE_INDEX]
        # Find correct index of security based on layout identified during initialisation to access expected payoff
        correct_index = list(self._market_ids.keys()).index(security)
        # Buying security
        if buy_units > 0 and self._settled_cash*DOLLAR_TO_CENTS > buy_price:
            order = Order(buy_price, 1, OrderType.LIMIT, OrderSide.BUY, self._market_ids[security],
                          ref=f"{security}_" + str(self._num_executed_trades[security] + 1))
            self.inform(f"Executing order: BUY {security} at {buy_price}")
            self.send_order(order)
            self._waiting_for_servers[security] = True
        # Could buy security but unable to due to low cash
        elif buy_units > 0 and self._settled_cash*DOLLAR_TO_CENTS < buy_price:
            self.inform(f"Not enough cash to react to trade: BUY {security} at {buy_price}")
        # Selling security
        if sell_units > 0 and self._market_holdings[correct_index] > 0:
            order = Order(sell_price, 1, OrderType.LIMIT, OrderSide.SELL, self._market_ids[security],
                          ref=f"{security}_" + str(self._num_executed_trades[security] + 1))
            self.inform(f"Executing order: SELL {security} at {sell_price}")
            self.send_order(order)
            self._waiting_for_servers[security] = True
        # Could sell security but unable to due to no available holdings
        elif sell_units > 0 and self._market_holdings[correct_index] < 1:
            self.inform(f"Not enough units to react to trade: SELL {security} at {sell_price}")

    def order_accepted(self, order):
        """
        Called when a submitted order is accepted by the server.
        :param order: order object sent to the server
        :return:
        """
        self.inform("Order ref {0} was accepted".format(order.ref))
        self._last_order_ref = order.ref
        # If order has been accepted, can increase the number of executed trades and reset the waiting for server flag
        # for this security
        for security in self._market_ids:
            if security in order.ref and "cancel" not in order.ref:
                self._num_executed_trades[security] += 1
                self._waiting_for_servers[security] = False

    def order_rejected(self, info, order):
        """
        Called when a submitted order is rejected by the server.
        :param info: contains error information from server
        :param order: order object sent to the server
        :return:
        """
        self.inform("Order ref {0} was rejected".format(order.ref))
        # If order has been rejected, can reset the waiting for server flag for this security
        for security in self._market_ids:
            # Expect to find security (string) in order.ref (also string) based on how order references were created in
            # _execute_strategy
            if security in order.ref:
                self._waiting_for_servers[security] = False

    def _cancel_open_trades(self, order_book, security):
        """
        Since bot only reacts based on best bid and ask prices, there should never be an open/pending order in the
        market. If for some reason trade is accepted but not settled, then open orders are cancelled.
        :param order_book: current order book of security
        :return:
        """
        for o in order_book:
            if o.mine and not self._waiting_for_servers[security]:
                cancel_order = copy.copy(o)
                cancel_order.type = OrderType.CANCEL
                cancel_order.ref = "cancel_pending_" + self._last_order_ref
                self.send_order(cancel_order)
                self._waiting_for_servers[security] = True

    def received_order_book(self, order_book, market_id):
        """
        Provides order book for a market.
        :param order_book: list of order objects
        :param market_id: id of a market (security)
        :return:
        """
        # Finding the security given market id
        for asset, m_id in self._market_ids.items():
            if m_id == market_id:
                security = asset
        # Cancel any open orders before checking and executing new set of orders
        self._cancel_open_trades(order_book, security)
        # If portfolio is not optimal and there are no orders waiting to be accepted/rejected for the security, execute
        # the trading strategy
        if not self.is_portfolio_optimal(order_book, security) and not self._waiting_for_servers[security]:
            self._execute_strategy(self._optimal_order, security)

    def received_holdings(self, holdings):
        """
        Captures available and settled cash and units of all securities
        :param holdings: dictionary containing available and settled cash and units
        :return:
        """
        try:
            self._market_holdings = np.array([])
            # Settled and available cash
            self._settled_cash = holdings["cash"]["cash"]/DOLLAR_TO_CENTS
            self._available_cash = holdings["cash"]["available_cash"]
            # Nested loop to ensure layout identified during initialisation is maintained. Finding settled units for
            # each security
            for security in self._market_ids:
                for market_id in holdings["markets"]:
                    if self._market_ids[security] == market_id:
                        self._market_holdings = np.append(self._market_holdings,
                                                          holdings["markets"][market_id]["units"])
        except Exception as e:
            self.error(e)

    def received_marketplace_info(self, marketplace_info):
        """
        Called when status of the associated marketplace switches between active (open for trading) and inactive (closed
        for trading). Reinitialises the instance of the class as holdings may have changed when marketplace reopens
        :param marketplace_info: dictionary with two keys, session_id and status
        :return:
        """
        session_id = marketplace_info["session_id"]
        if marketplace_info["status"]:
            self.inform("Marketplace is now open with session id: "+str(session_id))
        else:
            print("Marketplace is now closed")

    def run(self):
        self.initialise()
        self.start()


if __name__ == "__main__":
    FM_ACCOUNT = "teeming-truth"
    FM_EMAIL = "j.thomas19@student.unimelb.edu.au"
    FM_PASSWORD = "915951"
    MARKETPLACE_ID = 587

    bot = CAPMBot(FM_ACCOUNT, FM_EMAIL, FM_PASSWORD, MARKETPLACE_ID)
    bot.run()
