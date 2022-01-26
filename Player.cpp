#include "Player.h"
void Player::update(float delta) {

    playerpos.z += vz * delta;
}