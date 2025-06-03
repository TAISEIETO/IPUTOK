/**
* @file Game.h
* @brief �ۑ�9 : Game�N���X��`
* @author IT11H308 ok230112 �]���吰
* @date 2023/12/1
*/

#pragma once

#include <random>

#include "Enemy.h"

#define ENEMY_NUM_MIN 1							/* �G�̍ŏ��� */
#define ENEMY_NUM_MAX 10						/* �G�̍ő吔 */
#define ENEMY_HP_MIN 10							/* �G�̍ŏ�HP */
#define ENEMY_HP_MAX 50							/* �G�̍ő�HP */

#define ATTACK_MAX (10+1)						/* �U���̍ő�l */

class Game {
private:
	std::random_device rnd;						/* ���������p�I�u�W�F�N�g */
	int enemy_num;								/* �G�̑��� */
	Enemy* enemy[ENEMY_NUM_MAX];				/* �G�I�u�W�F�N�g */
	void initialize();/* ������ */
	void battle();/* �퓬 */
	void finalize();/* �I�� */

public:
	Game();/* �R���X�g���N�^ */
	void execute();/* ���s */
};