//
// Created by Volodymyr Avvakumov on 30.04.2026.
//

#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#include "httplib.h"

#define WORM 3
#define APPLE 2
#define WALL 1
#define EMPTY 0

struct pair_hash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// 10 x 10
std::vector<std::vector<short>> map = {
    {3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 1, 1, 0},
    {0, 1, 1, 1, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 1, 0, 1, 1, 0, 1, 0},
    {0, 1, 0, 1, 0, 0, 0, 0, 1, 0},
    {0, 1, 0, 0, 0, 1, 1, 0, 0, 0},
    {0, 1, 1, 1, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 1, 0, 0, 1, 0},
    {0, 1, 1, 1, 0, 1, 0, 0, 0, 2},
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0}
};

std::pair<int, int> worm_head_pose = {0, 0}; // {x, y}
std::pair<int, int> apple_pose = {9, 8};     // {x, y}

std::vector<std::pair<int, int>> findPath(
    const std::pair<int, int> worm_head_pose,
    const std::pair<int, int> apple_pose
) {
    std::queue<std::pair<int, int>> q;
    q.push(worm_head_pose);

    std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash> previous;
    previous[worm_head_pose] = {-1, -1};

    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();

        if (std::pair{x, y} == apple_pose) {
            break;
        }

        for (auto [dx, dy] : std::vector<std::pair<int, int>>{
            {-1, 0},
            {1, 0},
            {0, -1},
            {0, 1}
        }) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx < 0 || nx >= 10 || ny < 0 || ny >= 10) {
                continue;
            }

            if (map[ny][nx] == WALL) {
                continue;
            }

            if (previous.find({nx, ny}) != previous.end()) {
                continue;
            }

            q.emplace(nx, ny);
            previous[{nx, ny}] = {x, y};
        }
    }

    if (previous.find(apple_pose) == previous.end()) {
        return {};
    }

    std::vector<std::pair<int, int>> path;

    for (auto cur = apple_pose; cur != std::pair{-1, -1}; cur = previous[cur]) {
        path.push_back(cur);
    }

    std::reverse(path.begin(), path.end());

    return path;
}

std::pair<int, int> spawnApple() {
    std::vector<std::pair<int, int>> empty_cells;

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            if (map[y][x] == EMPTY) {
                empty_cells.push_back({x, y});
            }
        }
    }

    if (empty_cells.empty()) {
        return {-1, -1};
    }

    int index = rand() % empty_cells.size();
    auto [x, y] = empty_cells[index];

    map[y][x] = APPLE;

    return {x, y};
}

void initGame() {
    worm_head_pose = {0, 0};
    apple_pose = {9, 8};

    map[0][0] = WORM;
    map[8][9] = APPLE;
}

void stepGame() {
    std::vector<std::pair<int, int>> path = findPath(worm_head_pose, apple_pose);

    if (path.size() < 2) {
        return;
    }

    auto [old_x, old_y] = worm_head_pose;
    auto [new_x, new_y] = path[1];

    map[old_y][old_x] = EMPTY;

    worm_head_pose = {new_x, new_y};

    if (worm_head_pose == apple_pose) {
        map[new_y][new_x] = WORM;
        apple_pose = spawnApple();
    } else {
        map[new_y][new_x] = WORM;
    }
}

std::string to_string(int i) {
    if (i == WALL) {
        return "#";
    }
    if (i == APPLE) {
        return"";
    }
    if (i == WORM) {
        return"@";
    }
    return " ";
}

std::string mapToString() {
    std::string result;

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            result += to_string(map[y][x]);
            result += " ";
        }
        result += "\n";
    }

    return result;
}

int main() {
    srand(time(nullptr));

    initGame();

    httplib::Server server;

    server.Get("/step", [](const httplib::Request& req, httplib::Response& res) {
        stepGame();

        res.set_content(mapToString(), "text/plain");
    });

    server.Get("/map", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content(mapToString(), "text/plain");
    });

    std::cout << "Server started on http://localhost:8080\n";
    server.listen("0.0.0.0", 8080);

    return 0;
}