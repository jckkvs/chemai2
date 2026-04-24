import { createRouter, createWebHistory } from 'vue-router'
import DataUploadView from '../views/DataUploadView.vue'

const routes = [
  {
    path: '/',
    redirect: '/data'
  },
  {
    path: '/data',
    name: 'DataUpload',
    component: DataUploadView,
    meta: { title: 'データ読み込み' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
