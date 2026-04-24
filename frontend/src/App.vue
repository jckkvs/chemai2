<template>
  <el-container class="app-container">
    <!-- Sidebar -->
    <el-aside width="200px" class="sidebar">
      <div class="logo">
        <el-icon><Flask /></el-icon>
        <span>ChemAI Nexus</span>
      </div>
      <el-menu :default-active="activeMenu" router mode="vertical">
        <el-menu-item index="/data">
          <el-icon><Folder /></el-icon>
          <span>データ</span>
        </el-menu-item>
        <el-menu-item index="/eda">
          <el-icon><DataAnalysis /></el-icon>
          <span>EDA</span>
        </el-menu-item>
        <el-menu-item index="/ml">
          <el-icon><DataBoard /></el-icon>
          <span>機械学習</span>
        </el-menu-item>
        <el-menu-item index="/results">
          <el-icon><Document /></el-icon>
          <span>結果</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <!-- Main Content -->
    <el-container>
      <el-header class="app-header">
        <h2 class="text-xl font-bold">ChemAI Nexus v2.0</h2>
      </el-header>
      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Flask, Folder, DataAnalysis, DataBoard, Document } from '@element-plus/icons-vue'

const route = useRoute()
const activeMenu = computed(() => route.path)
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body, #app {
  height: 100%;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.app-container {
  height: 100%;
}

.sidebar {
  background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
  color: white;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 20px;
  font-size: 18px;
  font-weight: bold;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.el-menu {
  background: transparent;
  border-right: none;
}

.el-menu-item {
  color: rgba(255,255,255,0.8) !important;
}

.el-menu-item.is-active {
  background: rgba(255,255,255,0.1) !important;
  color: #42b983 !important;
}

.app-header {
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  padding: 0 20px;
  height: 60px;
}

.app-main {
  background: #f0f2f5;
  padding: 20px;
  overflow-y: auto;
}
</style>
