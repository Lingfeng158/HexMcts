#include <iostream>
#include <string>
#include <chrono>
#include <sys/time.h>
#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <cstdlib>
#include <cmath>
#include "jsoncpp/json.h"

// Class Headers

/**
 * @brief A helper struct for storing action in 2d
 * @NOTE: need to be hashable
 */
struct action2D
{
    int actionX;
    int actionY;

    // overloading operator== for unordered_map
    bool operator==(const action2D rhs) const
    {
        return actionX == rhs.actionX && actionY == rhs.actionY;
    }
};

template <>
struct std::hash<action2D>
{
    std::size_t operator()(action2D const &act) const noexcept
    {
        return std::hash<int>{}(act.actionX) ^ std::hash<int>{}(act.actionY);
    }
};

/**
 * @brief A helper struct for actionPrior
 *
 */
struct ActionPrior
{
    action2D action;
    float probability;
};

/**
 * @brief GameState class, representation of game states
 *
 */
class GameState
{
private:
    signed char board[11][11];
    int totalPieces;

public:
    /**
     * @brief Construct a new Game State object
     *
     */
    GameState();

    /**
     * @brief Determine if the game board is full
     *
     * @return true
     * @return false
     */
    bool boardIsFull();

    void setState(signed char b[][11]);

    /**
     * @brief Recover game state from server output
     *
     */
    void recoverState();

    /**
     * @brief if next player is red
     *
     * @return true
     * @return false
     */
    bool redPlaysNext();

    /**
     * @brief if last player is red
     *
     * @return true
     * @return false
     */
    bool redPlayedLast();

    /**
     * @brief next player plays action
     *
     * @param action the action next player takes
     * @return true
     * @return false
     */
    bool plays(int action);

    /**
     * @brief 2d action version of the same method "plays"
     *
     * @param x action//11
     * @param y action%11
     * @return true
     * @return false
     */
    bool plays(action2D action);

    /**
     * @brief check that given the current game state, if the game ends
     *
     * @return int 0: not end, 1: red ends, -1: black wins
     */
    int checkTermination();

    /**
     * @brief check if last player wins
     *
     * @return true last player won
     * @return false game not ending
     */
    bool lastPlayerWon();

    /**
     * @brief Check if one side has won
     *
     * @param isRed if the side to check is red player
     * @return true
     * @return false
     */
    bool oneSideTest(bool isRed);

    /**
     * @brief print the board of the game
     *
     */
    void printBoard();

    /**
     * @brief output legal play locations and prior heuristics
     *
     * @param forcedFirst the first hand is forced
     * @param forcedPlay forced play position
     */
    std::vector<ActionPrior> outputActionPrior(bool forcedFirst = true, action2D forcedPlay = {1, 2});
};

/**
 * @brief Node class for MCTS
 *
 */
class MCTSNode
{
private:
    MCTSNode *_parent;
    std::unordered_map<action2D, std::unique_ptr<MCTSNode>> _children;
    int _nVisits;
    float _quality;
    float _uct;
    float _heuristicFactor;
    bool _isRed;

public:
    MCTSNode(MCTSNode *node, float heuristic, bool isRed);
    /**
     * @brief expand a node, fill new nodes with action priors
     *
     */
    void expand(std::vector<ActionPrior> apPairs);

    /**
     * @brief evaluate a node based on UCT, quality, and heuristics
     *
     * @return float evaluation result
     */
    float evaluation(float xplorCoeff);

    std::unordered_map<action2D, std::unique_ptr<MCTSNode>> *getChildren();

    /**
     * @brief return if this node belongs to red player
     *
     * @return true
     * @return false
     */
    bool isRed();

    /**
     * @brief expose a node object by printing
     *
     */
    void expose();

    /**
     * @brief Set the Parent Null
     *
     */
    void setParentNull();

    /**
     * @brief greedily select a node based on exploration coefficient
     *
     * @param explorationCoeff exploration coefficient
     * @param isPlayout if the function is called during playout(during true play, uses another criterion)
     * @return int
     */
    std::unordered_map<action2D, std::unique_ptr<MCTSNode>>::iterator select(float xplorCoeff, bool isPlayout = true);

    /**
     * @brief update a node with returned result
     *
     * @param result result from rollout
     */
    void update(float result);

    /**
     * @brief recursively select parent until root and update
     *
     * @param result
     */
    void update_from_root(float result);

    /**
     * @brief if a node is leaf
     *
     * @return true
     * @return false
     */
    bool isLeaf();

    /**
     * @brief if a node is root
     *
     * @return true
     * @return false
     */
    bool isRoot();
};

class MCTS
{
private:
    std::unique_ptr<MCTSNode> _root;

    float _xplorCoeff;
    time_t _timeLimit;
    GameState _state;
    int _rolloutCounter;

public:
    /**
     * @brief Construct a new MCTS object
     *
     * @param explorationCoeff
     * @param startTime
     * @param timeLimit
     */
    MCTS(float explorationCoeff = 0.5, time_t timeLimit = 1000);

    /**
     * @brief Get the State object
     *
     * @return GameState
     */
    GameState getState();

    /**
     * @brief Set the State object
     *
     * @param state
     */
    void setState(GameState state);

    /**
     * @brief Get the Root object
     *
     * @return MCTSNode*
     */
    MCTSNode *getRoot();

    /**
     * @brief Get the Rollout Counter object
     *
     * @return int
     */
    int getRolloutCounter();

    MCTSNode *getNodeForAction(action2D action);

    /**
     * @brief Set the Root object
     *
     */
    void setRoot(MCTSNode *root);

    /**
     * @brief a wraper for gameState recovery from GameState class
     *
     */
    void gameStateRecover();

    /**
     * @brief playout until leaf node and do rollout
     *
     */
    void playout(GameState state_copy);

    /**
     * @brief naive rollout
     *
     * @param startNode
     * @param state
     */
    void singleRollout(MCTSNode *startNode, GameState state, int counter);

    /**
     * @brief branching every 32 actions, start with 2 branches,
     *
     * @param startNode
     * @param state
     * @param counter
     * @param search_indicator: normally 0 during non-branching, 1 for original thread, 2 for split thread
     */
    void branchingRollout(MCTSNode *startNode, GameState state, int counter, int search_indicator = 0);

    /**
     * @brief Get the next move through playout and rollout
     *
     * @return int representing next move
     */
    action2D getNextMove(time_t startTime, float timeMultiplier = 1.0);

    /**
     * @brief update MCTS internals with selected move
     *
     */
    void updateWithMove(action2D action);
};
//*********************************END of Headers

// Helper functions
Json::Value act2act(action2D action)
{
    Json::Value jAction;
    jAction["x"] = action.actionX;
    jAction["y"] = action.actionY;
    return jAction;
};

std::vector<action2D> findLinkedNodes(action2D action, std::vector<action2D> &pieceList)
{
    std::vector<action2D> result = std::vector<action2D>();
    int col = action.actionY;
    int row = action.actionX;
    if (col > 0)
    {
        action2D tmp = {row, col - 1};
        auto it = std::find(pieceList.begin(), pieceList.end(), tmp);
        if (it != pieceList.end())
        {
            result.push_back(tmp);
            pieceList.erase(it);
        }
    }
    if (col < 10)
    {
        action2D tmp = {row, col + 1};
        auto it = std::find(pieceList.begin(), pieceList.end(), tmp);
        if (it != pieceList.end())
        {
            result.push_back(tmp);
            pieceList.erase(it);
        }
    }
    if (row > 0)
    {
        action2D tmp = {row - 1, col};
        auto it = std::find(pieceList.begin(), pieceList.end(), tmp);
        if (it != pieceList.end())
        {
            result.push_back(tmp);
            pieceList.erase(it);
        }
    }

    if (row > 0 && col < 10)
    {
        action2D tmp = {row - 1, col + 1};
        auto it = std::find(pieceList.begin(), pieceList.end(), tmp);
        if (it != pieceList.end())
        {
            result.push_back(tmp);
            pieceList.erase(it);
        }
    }
    if (row < 10)
    {
        action2D tmp = {row + 1, col};
        auto it = std::find(pieceList.begin(), pieceList.end(), tmp);
        if (it != pieceList.end())
        {
            result.push_back(tmp);
            pieceList.erase(it);
        }
    }
    if (row < 10 && col > 0)
    {
        action2D tmp = {row + 1, col - 1};
        auto it = std::find(pieceList.begin(), pieceList.end(), tmp);
        if (it != pieceList.end())
        {
            result.push_back(tmp);
            pieceList.erase(it);
        }
    }
    return result;
};

/**
 * @brief DFS style search
 *
 * @param frontier list
 * @param pieceList list
 * @param isRed if the check is for red player
 * @return true checked player won
 * @return false checked player haven't won
 */
bool dfs(std::vector<action2D> &frontier, std::vector<action2D> &pieceList, bool isRed)
{
    if (frontier.size() == 0)
    {
        return false;
    }
    else
    {
        auto node = frontier.back();
        frontier.pop_back();
        if (isRed && node.actionX == 10)
        {
            return true;
        }
        if (!isRed && node.actionY == 10)
        {
            return true;
        }

        auto newFrontier = findLinkedNodes(node, pieceList);
        std::move(newFrontier.begin(), newFrontier.end(), std::back_inserter(frontier));
        return dfs(frontier, pieceList, isRed);
    }
}

std::array<char, 4> computeRangeBound(action2D action, int hexRange)
{
    char left_bound = std::max(0, action.actionX - hexRange);
    char right_bound = std::min(10, action.actionX + hexRange);
    char up_bound = std::max(0, action.actionY - hexRange);
    char bot_bound = std::min(10, action.actionY + hexRange);

    return {left_bound, right_bound, up_bound, bot_bound};
}

time_t getTimeInMilis()
{
    timeval time;
    gettimeofday(&time, NULL);
    time_t msecs_time = (time.tv_sec * 1000) + (time.tv_usec / 1000);
    return msecs_time;
}

//*************************End of Helper Functions

// Member function Impl
GameState::GameState() : board{}, totalPieces(0)
{
}

bool GameState::boardIsFull()
{
    return totalPieces == 121;
}

void GameState::recoverState()
{
    // 读入JSON
    std::string str;
    getline(std::cin, str);
    Json::Reader reader;
    Json::Value input;
    reader.parse(str, input);
    // 分析自己收到的输入和自己过往的输出，并恢复状态
    int turnID = input["responses"].size();
    for (int i = 0; i < turnID; i++)
    {
        plays({input["requests"][i]["x"].asInt(), input["requests"][i]["y"].asInt()});
        plays({input["responses"][i]["x"].asInt(), input["responses"][i]["y"].asInt()});
    }
    plays({input["requests"][turnID]["x"].asInt(), input["requests"][turnID]["y"].asInt()});
}

bool GameState::redPlaysNext()
{
    return totalPieces % 2 == 0;
}

bool GameState::redPlayedLast()
{
    return !redPlaysNext();
}

bool GameState::plays(int action)
{
    if (action >= 0 and action < 121)
    {
        int actionX = action / 11;
        int actionY = action % 11;
        action2D a2d = {actionX, actionY};
        return GameState::plays(a2d);
    }
    else
    {
        return false;
    }
}

bool GameState::plays(action2D action)
{
    if (action.actionX >= 0 && action.actionX < 11 && action.actionY >= 0 && action.actionY < 11)
    {
        board[action.actionX][action.actionY] = (totalPieces % 2 == 0) ? 1 : -1;
        totalPieces += 1;
        return true;
    }
    else
    {
        return false;
    }
}

bool GameState::oneSideTest(bool isRed)
{
    std::vector<action2D> pieceLocs;
    for (int i = 0; i < 11; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            if (board[i][j] == (isRed ? 1 : -1))
            {

                pieceLocs.push_back(action2D{i, j});
            }
        }
    }
    int piece_count = pieceLocs.size();
    if (piece_count < 11)
    {
        return false;
    }
    else
    {
        // find starting locations
        std::vector<action2D> startingLocs;
        std::vector<action2D> pieceList;
        for (auto it = pieceLocs.begin(); it != pieceLocs.end(); it++)
        {
            if ((isRed && it->actionX == 0) || (!isRed && it->actionY == 0))
            {
                startingLocs.push_back(*it);
            }
            else
            {
                pieceList.push_back(*it);
            }
        }
        if (startingLocs.size() == 0)
            return false;

        // std::copy_if(pieceLocs.begin(), pieceLocs.end(), std::back_inserter(startingLocs), [isRed](action2D loc) -> bool
        //              { return (isRed && loc.actionX == 0 || !isRed && loc.actionY == 0) ? true : false; });

        // for (int i = 1; i < 11; i++)
        // {
        //     int prevListSize = pieceList.size();
        //     std::copy_if(pieceLocs.begin(), pieceLocs.end(), std::back_inserter(startingLocs), [isRed, i](action2D loc) -> bool
        //                  { return (isRed && loc.actionX == i || !isRed && loc.actionY == i) ? true : false; });
        //     if (pieceList.size() - prevListSize == 0)
        //         return false;
        // }
        // std::for_each(pieceLocs.begin(), pieceLocs.end(), [](action2D action)
        //               { printf("action x:%d, y:%d\n", action.actionX, action.actionY); });
        return dfs(startingLocs, pieceList, isRed);
    }
}

bool GameState::lastPlayerWon()
{
    bool isRed = redPlayedLast();
    return oneSideTest(isRed);
}

int GameState::checkTermination()
{
    if (oneSideTest(true))
    {
        return 1;
    }
    else if (oneSideTest(false))
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

void GameState::setState(signed char b[][11])
{
    int counter = 0;
    for (int i = 0; i < 11; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            board[i][j] = b[i][j];
            if (b[i][j] != 0)
            {
                counter++;
            }
        }
    }
    totalPieces = counter;
}

void GameState::printBoard()
{
    for (int i = 0; i < 11; i++)
    {
        std::cout << std::string(i * 2, ' ');
        for (auto v : board[i])
        {
            std::cout.width(3);
            std::cout << (short)v;
        }
        std::cout << std::endl;
    }
}

std::vector<ActionPrior> GameState::outputActionPrior(bool forcedFirst, action2D forcedPlay)
{
    std::vector<action2D> actions;
    std::vector<ActionPrior> apPairs;
    if (totalPieces == 0 && forcedFirst)
    {
        actions.push_back(forcedPlay);
    }
    else
    {
        int range = 0;
        // force ignore border bound for first 10 moves,
        // force ignore border 2 rows for first 4 moves
        if (totalPieces <= 4)
        {
            range = 3;
        }
        else if (totalPieces <= 8)
        {
            range = 2;
        }
        else if (totalPieces <= 12)
        {
            range = 1;
        }
        for (int i = 0 + range; i < 11 - range; i++)
        {
            for (int j = 0 + range; j < 11 - range; j++)
            {
                if (board[i][j] == 0)
                {
                    actions.push_back(action2D{i, j});
                }
            }
        }
    }
    std::for_each(actions.begin(), actions.end(), [&](action2D action)
                  {
                      float heuristicMultiplier = 1.0;
                     // with in range 2, there is a piece, multiplier*1.1
                    //   auto [l, r, u, b] = computeRangeBound(action, 2);
                    //   if(isRed){
                    //       l += 1;
                    //       r -= 1;
                    //   }else{
                    //       u += 1;
                    //       b -= 1;
                    //   }
                    //   for (int i = u; i < b + 1; i++)
                    //   {
                    //       for (int j = l; j < r + 1; j++)
                    //       {
                    //           if (board[i][j] == (isRed?1:-1) && presenseIndicator==false)
                    //           {
                    //               heuristicMultiplier *= 1.2;
                    //               // break
                    //               i = b+1;
                    //               j = r+1;
                    //               presenseIndicator = true;
                    //           }
                    //       }
                    //   }
                    //   // bridge pattern multiplier*1.25
                     auto [l1, r1, u1, b1] = computeRangeBound(action, 1);
                      char blackPiece = 0, redPiece = 0;
                      for (int i = u1; i < b1 + 1; i++)
                      {
                          for (int j = l1; j < r1 + 1; j++)
                          {
                              if (board[i][j] == 1)
                              {
                                  redPiece += 1;
                              }
                              if (board[i][j] == -1)
                              {
                                  blackPiece += 1;
                              }
                          }
                      }
                      if ((redPiece >= 1 && blackPiece > 1) || (redPiece > 1 && blackPiece >= 1))
                      {
                          heuristicMultiplier *= 2;
                      }
                      if(redPiece+blackPiece>=4){
                          heuristicMultiplier *= 2;
                      }

                    //   // if close to border, multiplier*0.8
                    //   if (u1 * r1 == 0 || b1 == 10 || l1 == 10)
                    //   {
                    //       heuristicMultiplier *= 0.8;
                    //   }
                    if( action.actionX>=2 &&  action.actionX<=9 && action.actionY>=2 &&  action.actionY<=9){
                        heuristicMultiplier *= 1.5;
                    }
                      apPairs.push_back({action, heuristicMultiplier}); });
    return apPairs;
}

MCTSNode::MCTSNode(MCTSNode *node, float heuristic, bool isRed)
    : _parent(node), _children(), _nVisits(0), _quality(0), _uct(0), _heuristicFactor(heuristic), _isRed(isRed) {}

void MCTSNode::expand(std::vector<ActionPrior> apPairs)
{
    if (_children.size() == 0)
    {
        std::for_each(apPairs.begin(), apPairs.end(), [&](ActionPrior ap)
                      {
                          auto [action, prob] = ap;
                            // _children.insert({action, std::move(std::make_unique<MCTSNode>(MCTSNode(this, prob, !_isRed))) }); 
                            _children[action]=std::make_unique<MCTSNode>(MCTSNode(this, prob, !_isRed)); });
    }
    else
    {
        std::for_each(apPairs.begin(), apPairs.end(), [&](ActionPrior ap)
                      {
                      auto [action, prob] = ap;
                      auto it = _children.find(action);
                      if (it == _children.end())
                      {
                        //   _children[action]=std::make_unique<MCTSNode>(MCTSNode(this, prob, !_isRed));
                          _children.insert(std::make_pair(action, std::make_unique<MCTSNode>(MCTSNode(this, prob, !_isRed))));
                      } });
    }
}

float MCTSNode::evaluation(float xplorCoeff)
{
    _uct = _heuristicFactor * xplorCoeff * sqrt(2 * log(_parent->_nVisits)) /
           (1 + _nVisits);

    return _quality + _uct;
}

std::unordered_map<action2D, std::unique_ptr<MCTSNode>> *MCTSNode::getChildren()
{
    return &_children;
}

std::unordered_map<action2D, std::unique_ptr<MCTSNode>>::iterator MCTSNode::select(float xplorCoeff, bool isPlayout)
{
    if (_children.size() == 0)
    {
        return _children.end();
    }
    if (isPlayout)
    {
        return std::max_element(_children.begin(), _children.end(), [&xplorCoeff](const std::pair<const action2D, std::unique_ptr<MCTSNode>> &a, const std::pair<const action2D, std::unique_ptr<MCTSNode>> &b)
                                { return a.second.get()->evaluation(xplorCoeff) < b.second.get()->evaluation(xplorCoeff); });
    }
    else
    {
        return std::max_element(_children.begin(), _children.end(), [&xplorCoeff](const std::pair<const action2D, std::unique_ptr<MCTSNode>> &a, const std::pair<const action2D, std::unique_ptr<MCTSNode>> &b)
                                { return a.second.get()->_nVisits < b.second.get()->_nVisits; });
    }
}

void MCTSNode::update(float result)
{
    _nVisits += 1;
    _quality += (result - _quality) / _nVisits;
}

void MCTSNode::update_from_root(float result)
{
    if (_parent != nullptr)
    {
        _parent->update_from_root(-result * 0.95);
    }
    update(result);
}

bool MCTSNode::isRed()
{
    return _isRed;
}

void MCTSNode::expose()
{
    printf("Visit count: %d, qualiity: %f, uct: %f\n", _nVisits, _quality, _uct);
}

void MCTSNode::setParentNull()
{
    _parent = nullptr;
}

bool MCTSNode::isLeaf()
{
    return _children.size() == 0;
}

bool MCTSNode::isRoot()
{
    return _parent == nullptr;
}

MCTS::MCTS(float explorationCoeff, time_t timeLimit)
    : _root(new MCTSNode(nullptr, 1.0, false)), _xplorCoeff(explorationCoeff), _timeLimit(timeLimit), _state(GameState()), _rolloutCounter(0){};

GameState MCTS::getState()
{
    return _state;
}

void MCTS::setState(GameState state)
{
    _state = state;
    // need to update node red/black indicator accordingly
    _root = std::make_unique<MCTSNode>(MCTSNode(nullptr, 1.0, _state.redPlayedLast()));
}

MCTSNode *MCTS::getRoot()
{
    return _root.get();
}

void MCTS::setRoot(MCTSNode *root)
{
    _root = std::unique_ptr<MCTSNode>(root);
}

void MCTS::gameStateRecover()
{
    _state.recoverState();
}

void MCTS::playout(GameState state_copy)
{
    auto node = _root.get();
    while (true)
    {
        // printf("a1\n");
        if (node->isLeaf())
        {
            break;
        }
        auto it = node->select(_xplorCoeff);
        if (it == node->getChildren()->end())
        {
            printf("Error during playout select!");
            return;
        }
        else
        {
            state_copy.plays(it->first);
            node = it->second.get();
        }
    }
    // printf("a2\n");
    std::vector<ActionPrior> apList = state_copy.outputActionPrior();
    // std::vector<ActionPrior> apList;
    // apList.push_back({{9, 5}, 1.0});
    node->expand(apList);
    branchingRollout(node, state_copy, 0);
}

void MCTS::singleRollout(MCTSNode *startNode, GameState state, int counter)
{
    while (!state.boardIsFull())
    {
        if (counter <= 8)
        {
            float end = state.checkTermination();
            if (end != 0)
            {
                _rolloutCounter++;
                startNode->update_from_root(end * 10 / (counter + 1) * (startNode->isRed() ? 1 : -1));
                return;
            }
        }
        std::vector<ActionPrior> action_prior = state.outputActionPrior();
        auto it = std::max_element(action_prior.begin(), action_prior.end(), [](const ActionPrior &ap1, const ActionPrior &ap2)
                                   { return ap1.probability < ap2.probability; });
        action2D action = it->action;
        state.plays(action);
        counter++;
    }
    int end = state.checkTermination();
    if (end != 0)
    {
        _rolloutCounter++;
        startNode->update_from_root(end * (startNode->isRed() ? 1 : -1));
        return;
    }
    else
    {
        printf("Error at the end of singleRollout!\n");
    }
}

int MCTS::getRolloutCounter()
{
    return _rolloutCounter;
}

MCTSNode *MCTS::getNodeForAction(action2D action)
{
    auto it = _root->getChildren()->find(action);
    if (it != _root->getChildren()->end())
    {
        return it->second.get();
    }
    return nullptr;
}

void MCTS::branchingRollout(MCTSNode *startNode, GameState state, int counter, int search_indicator)
{
    // 32 diviser for branching
    int diviser = 31;
    // 16 diviser for termination check
    int termDiv = 15;
    while (!state.boardIsFull())
    {
        // printf("a1\n");
        // check for termination for first 5 rounds
        // for 10 immediate step, the closer to startNode, the higher the reward
        if (counter <= 10)
        {
            float end = state.checkTermination();
            if (end != 0)
            {
                startNode->update_from_root(end * 16 / (counter + 1) * (startNode->isRed() ? 1 : -1));
                return;
            }
        }

        if ((counter & termDiv) == 0 && counter != 0)
        {
            float end = state.checkTermination();
            if (end != 0)
            {
                _rolloutCounter++;

                startNode->update_from_root(end * (startNode->isRed() ? 1 : -1) * pow(0.995, counter));
                return;
            }
        }

        // branching
        if (search_indicator == 0 && (counter & diviser) == 0)
        {

            search_indicator = 1;
            GameState stateCopy = state;
            branchingRollout(startNode, stateCopy, counter, 2);
        }

        std::vector<ActionPrior> action_prior = state.outputActionPrior();
        action2D action;
        int mid = action_prior.size() / 2;
        if (search_indicator == 0)
        {
            auto it = std::max_element(action_prior.begin(), action_prior.end(), [](const ActionPrior &ap1, const ActionPrior &ap2)
                                       { return ap1.probability < ap2.probability; });
            action = it->action;
        }
        else if (search_indicator == 1 && action_prior.size() > 1)
        {
            // first half of action space
            auto it = std::max_element(action_prior.begin(), action_prior.begin() + mid, [](const ActionPrior &ap1, const ActionPrior &ap2)
                                       { return ap1.probability < ap2.probability; });
            action = it->action;
        }
        else
        {
            // second half of action space
            auto it = std::max_element(action_prior.begin() + mid, action_prior.end(), [](const ActionPrior &ap1, const ActionPrior &ap2)
                                       { return ap1.probability < ap2.probability; });
            action = it->action;
        }

        state.plays(action);
        counter++;
        search_indicator = 0;
    }
    int end = state.checkTermination();
    if (end != 0)
    {
        _rolloutCounter++;
        startNode->update_from_root(end * (startNode->isRed() ? 1 : -1) * pow(0.995, counter));
        return;
    }
    else
    {
        printf("Error at the end of branchingRollout!\n");
    }
}

action2D MCTS::getNextMove(time_t startTime, float timeMultiplier)
{
    float timeLim = _timeLimit * timeMultiplier;
    time_t time_passes = 0;
    while (((1.0 * time_passes / timeLim) * 100) < 87)
    {
        for (int i = 0; i < 50; i++)
        {
            auto stateCopy = _state;
            playout(stateCopy);
        }

        time_passes = getTimeInMilis() - startTime;
    }
    auto it = _root->select(_xplorCoeff, false);
    if (it == _root->getChildren()->end())
    {
        printf("Error at getNextMove select!");
        return {0, 0};
    }
    else
    {
        return it->first;
    }
}

void MCTS::updateWithMove(action2D action)
{
    auto rootChildren = _root->getChildren();
    auto it = rootChildren->find(action);
    if (it != rootChildren->end())
    {
        _state.plays(action);
        _root = std::move(it->second);
        _root->setParentNull();
    }
    else
    {
        _state.plays(action);
        _root = std::make_unique<MCTSNode>(MCTSNode(nullptr, 1.0, _state.redPlayedLast()));
    }
    // _state.plays(action);
    // _root = std::make_unique<MCTSNode>(MCTSNode(nullptr, 1.0, _state.redPlayedLast()));
}
// int main()
// {
//     MCTS mcts;
//     time_t startTime = getTimeInMilis();
//     mcts.getNextMove(startTime);
//     std::cout << mcts.getRolloutCounter();
// }

int main()
{

    time_t startTime = getTimeInMilis();

    GameState g;
    g.recoverState();
    MCTS mcts;
    mcts.setState(g);
    action2D action = mcts.getNextMove(startTime, 1.9);
    mcts.updateWithMove(action);

    Json::Value ret;
    ret["response"] = act2act(action);
    Json::FastWriter writer;
    Json::Reader reader;
    Json::Value input;
    std::string str;
    std::cout << writer.write(ret) << std::endl;
    std::cout << ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<" << std::endl;
    fflush(stdout);
    while (true)
    {

        getline(std::cin, str);
        startTime = getTimeInMilis();
        reader.parse(str, input);
        action = {input["x"].asInt(), input["y"].asInt()};
        mcts.updateWithMove(action);
        action = mcts.getNextMove(startTime);
        mcts.updateWithMove(action);
        ret["response"] = act2act(action);
        std::cout << writer.write(ret) << std::endl;
        std::cout << ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<" << std::endl;
        fflush(stdout);
    }
};
