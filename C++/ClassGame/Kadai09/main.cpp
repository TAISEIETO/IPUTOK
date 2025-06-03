/**
* @file main.cpp
* @brief 課題9 : メイン関数
* @author IT11H308 ok230112 江藤大晴
* @date 2023/12/1
*/

#include "Game.h"

int main() {

	/* ゲームオブジェクトを作成 */
	Game game;

	/* ゲームを実行 */
	game.execute();

	rewind(stdin);
	getchar();

	return 0;
}